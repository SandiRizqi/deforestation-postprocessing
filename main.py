import rasterio
import numpy as np
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy.ndimage import binary_opening, binary_closing, label, binary_dilation
from scipy import ndimage
import pandas as pd

def extract_date_from_filename(filename):
    """
    Extract tahun dan bulan dari nama file dengan format: nama_tahun_bulan
    Contoh: AOI_2019_01 -> (2019, 1) atau "201901"

    Returns:
        tuple: (year, month) atau string "YYYYMM"
    """
    match = re.search(r'(\d{4})_(\d{2})', filename)
    if match:
        year = match.group(1)
        month = match.group(2)
        return f"{year}{month}"
    else:
        raise ValueError(f"Format filename tidak sesuai: {filename}. Gunakan format: nama_YYYY_MM")

def morphological_cleanup(binary_map, kernel_size=5, operation='closing'):
    """
    Lakukan morphological opening/closing untuk eliminate noise.

    Args:
        binary_map: binary array (0 or 1)
        kernel_size: ukuran kernel untuk morphological operation
        operation: 'opening' (remove small noise), 'closing' (fill small gaps), 'both'

    Returns:
        cleaned binary map
    """
    # Buat kernel (struktur elemen)
    kernel = ndimage.generate_binary_structure(2, 2)

    cleaned = binary_map.copy()

    if operation in ['opening', 'both']:
        # Opening: Erosion diikuti dilation - remove small objects
        cleaned = binary_opening(cleaned, structure=kernel)

    if operation in ['closing', 'both']:
        # Closing: Dilation diikuti erosion - fill small holes
        cleaned = binary_closing(cleaned, structure=kernel)

    return cleaned.astype(int)

def enforce_spatial_coherence(deforestation_binary, confidence_map,
                             min_cluster_sizes=None, neighborhood_size=5):
    """
    Enforce spatial coherence: remove isolated pixels dan small patches.
    Pixels harus dalam cluster untuk maintain/increase confidence.

    Args:
        deforestation_binary: binary map deforestasi (0 or 1)
        confidence_map: confidence level map (0-4)
        min_cluster_sizes: dict {confidence_level: min_pixels}
                          Default: {1: 50, 2: 50, 3: 100, 4: 200}
        neighborhood_size: size kernel untuk neighborhood check (default 5x5)

    Returns:
        tuple: (filtered_binary, filtered_confidence)
    """
    if min_cluster_sizes is None:
        min_cluster_sizes = {
            1: 50,    # Level 1: minimum 50 pixels cluster
            2: 50,    # Level 2: minimum 50 pixels cluster
            3: 100,   # Level 3: minimum 100 pixels cluster
            4: 200    # Level 4: minimum 200 pixels cluster (mature defo)
        }

    # Label connected components
    labeled_array, num_features = label(deforestation_binary)
    unique_labels, counts = np.unique(labeled_array, return_counts=True)

    filtered_binary = deforestation_binary.copy()
    filtered_confidence = confidence_map.copy()

    # Untuk setiap cluster, check apakah memenuhi minimum size
    for idx, label_id in enumerate(unique_labels):
        if label_id == 0:  # Background
            continue

        cluster_size = counts[idx]
        cluster_mask = (labeled_array == label_id)

        # Dapatkan confidence level dari cluster ini
        cluster_confidences = confidence_map[cluster_mask]
        max_confidence = cluster_confidences.max()

        # Check minimum size requirement
        min_required = min_cluster_sizes.get(max_confidence, 50)

        if cluster_size < min_required:
            # Cluster terlalu kecil → remove
            filtered_binary[cluster_mask] = 0
            filtered_confidence[cluster_mask] = 0

    # Additional: Check neighborhood coherence untuk isolated pixels
    # Pixel tanpa neighbors dengan confidence < 3 → penalize
    kernel = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    neighborhood_count = ndimage.convolve(
        filtered_binary.astype(float),
        kernel.astype(float),
        mode='constant',
        cval=0
    ).astype(int)

    height, width = filtered_binary.shape
    for i in range(height):
        for j in range(width):
            if filtered_binary[i, j] == 0:
                continue

            conf = filtered_confidence[i, j]
            neighbors = neighborhood_count[i, j] - 1  # Exclude self

            # Pixel terisolasi (0-1 neighbors) dengan low confidence → remove
            if conf < 3 and neighbors < 2:
                filtered_binary[i, j] = 0
                filtered_confidence[i, j] = 0

    return filtered_binary, filtered_confidence

def classify_deforestation(raster_data, date_code, threshold=0.7):
    """
    Klasifikasi raster deforestasi berdasarkan threshold.

    Args:
        raster_data: numpy array dengan nilai 0-1
        date_code: string format YYYYMM dari filename
        threshold: nilai threshold (default 0.7)

    Returns:
        numpy array dengan nilai 0 atau kode1YYYYMM (1 = confidence level 1)
    """
    classified = np.zeros_like(raster_data, dtype=np.int32)

    # Area dengan nilai > threshold menjadi 1YYYYMM (confidence 1 = baru terdeteksi)
    deforest_mask = raster_data > threshold
    classified[deforest_mask] = int(f"1{date_code}")

    return classified

def build_temporal_history(folder_path):
    """
    Baca semua file .tif dari folder dan organize by month untuk persistence analysis

    Args:
        folder_path: path ke folder dengan file .tif

    Returns:
        dict: {month_key: {'path': file_path, 'data': raster_data, 'date_code': YYYYMM}}
    """
    folder = Path(folder_path)
    tif_files = sorted(folder.glob('*.tif'))

    temporal_history = {}

    for tif_file in tif_files:
        try:
            date_code = extract_date_from_filename(tif_file.stem)

            with rasterio.open(tif_file) as src:
                data = src.read(1).astype(np.float32)

            temporal_history[date_code] = {
                'path': str(tif_file),
                'data': data,
                'date_code': date_code
            }

            print(f"Loaded: {tif_file.name} (Date code: {date_code})")

        except Exception as e:
            print(f"Skip {tif_file.name}: {e}")

    return temporal_history

def calculate_persistence_confidence(deforestation_map, date_code, temporal_history,
                                     persistence_months=6, threshold=0.7):
    """
    Hitung confidence level berdasarkan temporal persistence.
    Dengan morphological filtering dan spatial coherence enforcement.

    Logika:
    - Level 1: Deforestasi hanya terdeteksi di current month (musiman/noise)
    - Level 2: Konsisten 2 bulan
    - Level 3: Konsisten 3-5 bulan
    - Level 4: Konsisten ≥persistence_months (permanen deforestasi)

    Args:
        deforestation_map: binary map deforestasi dari current month (0-1)
        date_code: YYYYMM dari bulan current
        temporal_history: dict history dari build_temporal_history()
        persistence_months: minimum bulan untuk dianggap permanen
        threshold: threshold untuk deteksi deforestasi

    Returns:
        tuple: (conf_score_map, confidence_map)
    """

    # Dapatkan urutan bulan
    sorted_dates = sorted(temporal_history.keys())
    current_idx = sorted_dates.index(date_code)

    # Binary deforestation map untuk current month
    current_defo_binary = (deforestation_map > threshold).astype(int)

    # Morphological cleanup untuk remove noise
    current_defo_cleaned = morphological_cleanup(current_defo_binary, kernel_size=5, operation='both')

    # Inisialisasi confidence map
    confidence_map = np.zeros_like(deforestation_map, dtype=int)

    height, width = current_defo_cleaned.shape

    # Untuk setiap pixel, hitung berapa bulan ke belakang konsisten deforestasi
    for i in range(height):
        for j in range(width):
            # Skip jika pixel tidak deforestasi di current month
            if current_defo_cleaned[i, j] == 0:
                continue

            # Hitung persistensi (berapa bulan ke belakang konsisten)
            consecutive_months = 1  # Current month

            # Cek bulan-bulan sebelumnya (backward in time)
            for k in range(1, min(persistence_months, current_idx + 1)):
                prev_idx = current_idx - k
                prev_date = sorted_dates[prev_idx]
                prev_data = temporal_history[prev_date]['data']

                # Check jika pixel juga deforestasi di bulan sebelumnya
                if prev_data[i, j] > threshold:
                    consecutive_months += 1
                else:
                    break  # Stop jika tidak konsisten

            # Map consecutive_months ke confidence level
            if consecutive_months == 1:
                conf_level = 1  # Hanya current month → LOW
            elif consecutive_months == 2:
                conf_level = 2  # 2 bulan
            elif consecutive_months >= persistence_months:
                conf_level = 4  # Full persistence → HIGH (permanen)
            else:
                conf_level = 3  # 3-5 bulan → MEDIUM

            confidence_map[i, j] = conf_level

    # Enforce spatial coherence
    final_binary, final_confidence = enforce_spatial_coherence(
        current_defo_cleaned,
        confidence_map,
        min_cluster_sizes={1: 50, 2: 50, 3: 100, 4: 200},
        neighborhood_size=5
    )

    # Generate raster output dengan format [C][YYYYMM]
    conf_score_map = np.where(
        final_confidence > 0,
        final_confidence * 1000000 + int(date_code),
        0
    ).astype(np.int32)

    return conf_score_map, final_confidence

def analyze_recovery_pattern(previous_data, temporal_history, current_date_code,
                            threshold=0.7, persistence_months=6):
    """
    Analisis pixel yang sebelumnya terdeteksi deforestasi tapi sekarang tidak.
    Tentukan apakah confidence harus diturunkan atau reset ke 0.

    Logika:
    - Jika deforestasi terdeteksi lagi dalam beberapa bulan → turunkan confidence 1 level
    - Jika hilang >persistence_months tanpa deteksi → reset ke 0 (anggap recovery/false positive)
    - Track berapa bulan pixel "missing" dari pattern deforestasi

    Returns:
        dict dengan recovery analysis untuk setiap pixel
    """
    # Dapatkan urutan bulan
    sorted_dates = sorted(temporal_history.keys())
    current_idx = sorted_dates.index(current_date_code)

    recovery_status = {}

    height, width = previous_data.shape

    for i in range(height):
        for j in range(width):
            if previous_data[i, j] == 0:
                continue  # Skip pixel tanpa deforestasi sebelumnya

            old_value = previous_data[i, j]
            old_value_str = str(old_value)
            old_conf_digit = int(old_value_str[0])
            original_date_code = old_value_str[1:]

            # Cek berapa bulan terakhir pixel ini "hilang" dari deteksi
            months_without_detection = 0

            # Cek bulan-bulan sebelumnya (backward)
            for k in range(1, min(persistence_months + 1, current_idx + 1)):
                prev_idx = current_idx - k
                prev_date = sorted_dates[prev_idx]
                prev_data = temporal_history[prev_date]['data']

                if prev_data[i, j] <= threshold:  # Tidak terdeteksi deforestasi
                    months_without_detection += 1
                else:
                    break  # Stop jika ada deteksi

            recovery_status[(i, j)] = {
                'old_conf': old_conf_digit,
                'months_missing': months_without_detection,
                'original_date': original_date_code,
                'should_reset': months_without_detection >= persistence_months,
                'should_decrement': months_without_detection > 0 and months_without_detection < persistence_months
            }

    return recovery_status

def create_initial_deforestation_raster(raw_image, output_path=None, threshold=0.7,
                                        temporal_history=None, persistence_months=6):
    """
    Buat raster deforestasi awal dengan confidence level berdasarkan persistence.
    Gunakan untuk membuat raster pertama kali.

    Args:
        raw_image: path ke raster raw (nilai 0-1)
        output_path: path untuk simpan hasil (opsional)
        threshold: threshold untuk deteksi deforestasi
        temporal_history: dict dari build_temporal_history() untuk persistence analysis
        persistence_months: minimum bulan untuk dianggap permanen

    Returns:
        tuple: (confidence_score_array, output_path)
    """
    filename = Path(raw_image).stem
    date_code = extract_date_from_filename(filename)

    with rasterio.open(raw_image) as src:
        raw_data = src.read(1).astype(np.float32)
        profile = src.profile

    # Jika ada temporal history, hitung confidence dengan persistence
    if temporal_history is not None:
        conf_score_map, conf_map = calculate_persistence_confidence(
            raw_data, date_code, temporal_history,
            persistence_months=persistence_months,
            threshold=threshold
        )

        # Print statistik
        print(f"\nInitial Deforestation Raster: {date_code}")
        print(f"  Level 1 (LOW):    {(conf_map == 1).sum()} pixels (musiman)")
        print(f"  Level 2:          {(conf_map == 2).sum()} pixels")
        print(f"  Level 3 (MED):    {(conf_map == 3).sum()} pixels")
        print(f"  Level 4 (HIGH):   {(conf_map == 4).sum()} pixels (permanen)")
    else:
        # Jika tidak ada history, gunakan confidence default 1
        binary_defo = (raw_data > threshold).astype(int)
        binary_defo = morphological_cleanup(binary_defo, kernel_size=5, operation='both')

        conf_score_map = np.where(
            binary_defo > 0,
            1 * 1000000 + int(date_code),
            0
        ).astype(np.int32)

        print(f"\nInitial Deforestation Raster: {date_code}")
        print(f"  Total deforestation pixels: {(binary_defo > 0).sum()}")
        print(f"  Note: Confidence level 1 (first detection)")

    if output_path is None:
        output_path = f"deforestation_{date_code}.tif"

    profile.update(dtype=rasterio.int32)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(conf_score_map, 1)

    print(f"Saved: {output_path}\n")

    return conf_score_map, output_path

def update_deforestation_data(previous_image, after_image, temporal_history=None,
                              output_path=None, threshold=0.7, persistence_months=6):
    """
    Update raster deforestasi dengan persistence-based confidence scoring.
    Dengan morphological filtering dan spatial coherence enforcement.

    Logika:
    - Pixel baru terdeteksi deforestasi → mulai dari confidence 1
    - Pixel yang tetap deforestasi → increment confidence (1→2→3→4)
    - Pixel yang hilang dari deteksi beberapa bulan → decrement confidence (4→3→2→1)
    - Pixel yang hilang >persistence_months → reset ke 0 (recovery/false positive)
    - Confidence maksimal = 4 (permanen deforestasi)
    - Tanggal awal terdeteksi tetap dipertahankan (kecuali reset)

    Args:
        previous_image: path ke raster deforestasi sebelumnya (sudah diklasifikasi)
        after_image: path ke raster terbaru (raw data dengan nilai 0-1)
        temporal_history: dict dari build_temporal_history() untuk persistence analysis
        output_path: path untuk simpan hasil (opsional)
        threshold: threshold untuk deteksi deforestasi
        persistence_months: minimum bulan untuk dianggap permanen

    Returns:
        tuple: (updated_confidence_array, output_path)
    """

    after_filename = Path(after_image).stem
    current_date_code = extract_date_from_filename(after_filename)

    # Baca raster sebelumnya (sudah dengan confidence score)
    with rasterio.open(previous_image) as src:
        previous_data = src.read(1)
        profile = src.profile

    # Baca raster sesudah (raw data)
    with rasterio.open(after_image) as src:
        after_data = src.read(1).astype(np.float32)

    # Validasi ukuran raster
    if previous_data.shape != after_data.shape:
        raise ValueError(
            f"Ukuran raster tidak sama!\n"
            f"  Previous: {previous_data.shape}\n"
            f"  After: {after_data.shape}\n"
            f"Pastikan kedua raster memiliki dimensi yang sama."
        )

    # Binary deforestation dari data terbaru
    after_defo_binary = (after_data > threshold).astype(int)

    # Hitung confidence level jika ada temporal history
    if temporal_history is not None:
        conf_score_current, conf_map_current = calculate_persistence_confidence(
            after_data, current_date_code, temporal_history,
            persistence_months=persistence_months,
            threshold=threshold
        )

        # Analisis pixel yang sebelumnya deforestasi tapi sekarang tidak
        recovery_analysis = analyze_recovery_pattern(
            previous_data, temporal_history, current_date_code,
            threshold=threshold,
            persistence_months=persistence_months
        )
    else:
        # Fallback: gunakan confidence 1 untuk semua pixel baru
        after_defo_binary = morphological_cleanup(after_defo_binary, kernel_size=5, operation='both')
        conf_score_current = np.where(
            after_defo_binary > 0,
            1 * 1000000 + int(current_date_code),
            0
        ).astype(np.int32)
        recovery_analysis = {}

    # Updated logic
    updated = previous_data.copy().astype(np.int32)

    height, width = updated.shape

    # Validasi ulang ukuran
    if after_defo_binary.shape != (height, width):
        raise ValueError(
            f"Mismatch ukuran setelah processing!\n"
            f"  Expected: {(height, width)}\n"
            f"  Got: {after_defo_binary.shape}"
        )

    # Tracking statistik
    new_defo_count = 0
    existing_defo_count = 0
    confidence_increased = 0
    confidence_decreased = 0
    defo_reset_to_forest = 0

    for i in range(height):
        for j in range(width):
            try:
                current_has_defo = after_defo_binary[i, j] > 0
                previous_has_defo = previous_data[i, j] > 0
            except IndexError as e:
                print(f"ERROR at position [{i}, {j}]")
                print(f"  after_defo_binary shape: {after_defo_binary.shape}")
                print(f"  previous_data shape: {previous_data.shape}")
                print(f"  updated shape: {updated.shape}")
                raise

            if current_has_defo and not previous_has_defo:
                # KASUS 1: Pixel baru terdeteksi deforestasi
                updated[i, j] = conf_score_current[i, j]
                new_defo_count += 1

            elif current_has_defo and previous_has_defo:
                # KASUS 2: Pixel sudah deforestasi sebelumnya dan tetap deforestasi
                old_value = previous_data[i, j]
                old_value_str = str(old_value)

                # Extract confidence digit dan date code lama
                old_conf_digit = int(old_value_str[0])
                original_date_code = old_value_str[1:]

                # Increment confidence digit (1→2→3→4), maksimal 4
                new_conf_digit = min(old_conf_digit + 1, 4)

                # Buat nilai baru
                new_value = int(f"{new_conf_digit}{original_date_code}")
                updated[i, j] = new_value
                existing_defo_count += 1
                confidence_increased += 1

            elif not current_has_defo and previous_has_defo:
                # KASUS 3: Pixel yang sebelumnya deforestasi tapi sekarang tidak
                recovery_info = recovery_analysis.get((i, j))

                if recovery_info is None:
                    updated[i, j] = 0
                    defo_reset_to_forest += 1

                elif recovery_info['should_reset']:
                    updated[i, j] = 0
                    defo_reset_to_forest += 1

                elif recovery_info['should_decrement']:
                    old_conf = recovery_info['old_conf']
                    original_date = recovery_info['original_date']

                    new_conf_digit = max(old_conf - 1, 0)

                    if new_conf_digit == 0:
                        updated[i, j] = 0
                        defo_reset_to_forest += 1
                    else:
                        new_value = int(f"{new_conf_digit}{original_date}")
                        updated[i, j] = new_value

                    confidence_decreased += 1

    # Apply morphological filtering untuk remove noise
    updated_binary = (updated > 0).astype(int)
    updated_cleaned = morphological_cleanup(updated_binary, kernel_size=5, operation='both')

    # Extract confidence levels dari updated raster
    updated_confidence = np.zeros_like(updated)
    for conf_level in range(1, 5):
        mask = ((updated // 1000000) == conf_level)
        updated_confidence[mask] = conf_level

    # Enforce spatial coherence
    final_binary, final_confidence = enforce_spatial_coherence(
        updated_cleaned,
        updated_confidence,
        min_cluster_sizes={1: 10, 2: 50, 3: 100, 4: 200},
        neighborhood_size=5
    )

    # Reconstruct final raster dengan confidence + date code
    final_raster = np.zeros_like(updated)
    for i in range(height):
        for j in range(width):
            if final_confidence[i, j] > 0:
                old_value_str = str(updated[i, j])
                if len(old_value_str) > 1:
                    original_date = old_value_str[1:]
                    final_raster[i, j] = int(f"{final_confidence[i, j]}{original_date}")
                else:
                    final_raster[i, j] = int(f"{final_confidence[i, j]}{current_date_code}")

    updated = final_raster.astype(np.int32)

    # Print statistik update
    print(f"\nUpdate Deforestation Raster: {current_date_code}")
    print(f"  New deforestation detected:        {new_defo_count} pixels")
    print(f"  Existing deforestation (both):     {existing_defo_count} pixels")
    print(f"    → Confidence increased:          {confidence_increased} pixels")
    print(f"  Previous deforestation (missing):  {defo_reset_to_forest + confidence_decreased} pixels")
    print(f"    → Confidence decreased:          {confidence_decreased} pixels")
    print(f"    → Reset to forest (0):           {defo_reset_to_forest} pixels")
    print(f"  After spatial coherence filter:")
    print(f"    → Level 1 pixels (isolated removed): {(final_confidence == 1).sum()} px")
    print(f"    → Level 2 pixels (clustered):       {(final_confidence == 2).sum()} px")
    print(f"    → Level 3 pixels (medium conf):     {(final_confidence == 3).sum()} px")
    print(f"    → Level 4 pixels (confirmed defo):  {(final_confidence == 4).sum()} px")

    # Generate output path jika tidak diberikan
    if output_path is None:
        output_path = f"deforestation_updated_{current_date_code}.tif"

    # Update profile untuk int32
    profile.update(dtype=rasterio.int32)

    # Simpan hasil
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(updated, 1)

    print(f"Saved: {output_path}\n")

    return updated, output_path

def output_file_exists(output_folder, date_code):
    """
    Cek apakah file output sudah ada di output folder
    
    Args:
        output_folder: path ke output folder
        date_code: YYYYMM format
        
    Returns:
        bool: True jika file sudah ada, False sebaliknya
    """
    output_path = Path(output_folder) / f"deforestation_{date_code}.tif"
    return output_path.exists()

def process_batch_deforestation(input_folder, output_folder, threshold=0.7, persistence_months=6):
    """
    Process semua file .tif secara batch dengan persistence-based confidence.
    Dengan morphological filtering dan spatial coherence enforcement.
    Skip file yang sudah diproses untuk efisiensi.

    Args:
        input_folder: folder dengan file raw prediction (0-1)
        output_folder: folder untuk menyimpan hasil
        threshold: threshold deteksi deforestasi
        persistence_months: minimum bulan untuk dianggap permanen
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Build temporal history dari semua file
    print("=" * 100)
    print("BUILDING TEMPORAL HISTORY")
    print("=" * 100)
    temporal_history = build_temporal_history(input_folder)

    if not temporal_history:
        print("No files found!")
        return

    sorted_dates = sorted(temporal_history.keys())
    print(f"\nTotal files loaded: {len(temporal_history)}")
    print(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}\n")

    # Process pertama kali (create initial raster)
    print("=" * 100)
    print("PROCESSING INITIAL RASTER")
    print("=" * 100)
    first_date = sorted_dates[0]
    first_file_path = temporal_history[first_date]['path']

    # Cek apakah file output sudah ada
    if output_file_exists(output_folder, first_date):
        print(f"✓ Output untuk {first_date} sudah ada, SKIP")
        output_file = output_path / f"deforestation_{first_date}.tif"
    else:
        output_file = output_path / f"deforestation_{first_date}.tif"
        create_initial_deforestation_raster(
            first_file_path,
            output_path=str(output_file),
            threshold=threshold,
            temporal_history=temporal_history,
            persistence_months=persistence_months
        )

    # Update untuk bulan-bulan berikutnya
    print("=" * 100)
    print("PROCESSING SUBSEQUENT MONTHS")
    print("=" * 100)

    previous_output = output_file

    for current_date in sorted_dates[1:]:
        # Cek apakah file output sudah ada
        if output_file_exists(output_folder, current_date):
            print(f"✓ Output untuk {current_date} sudah ada, SKIP")
            output_file = output_path / f"deforestation_{current_date}.tif"
            previous_output = output_file
            continue

        current_file_path = temporal_history[current_date]['path']
        output_file = output_path / f"deforestation_{current_date}.tif"

        update_deforestation_data(
            str(previous_output),
            current_file_path,
            temporal_history=temporal_history,
            output_path=str(output_file),
            threshold=threshold,
            persistence_months=persistence_months
        )

        previous_output = output_file

    print("=" * 100)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 100)
    print(f"Output saved to: {output_path}")


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Deforestation monitoring dengan persistence-based confidence scoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --input-folder ../../ --output-folder ./
  python main.py --input-folder ../../ --output-folder ./ --threshold 0.9 --persistence 6
  python main.py -i ../../ -o ./ -t 0.85 -p 6
        '''
    )
    
    parser.add_argument(
        '--input-folder', '-i',
        required=True,
        type=str,
        help='Folder path untuk input .tif files'
    )
    
    parser.add_argument(
        '--output-folder', '-o',
        required=True,
        type=str,
        help='Folder path untuk output hasil'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        default=0.7,
        type=float,
        help='Threshold untuk deteksi deforestasi (default: 0.7)'
    )
    
    parser.add_argument(
        '--persistence', '-p',
        default=6,
        type=int,
        help='Minimum bulan untuk dianggap permanen deforestasi (default: 6)'
    )
    
    return parser.parse_args()


def validate_paths(input_folder, output_folder):
    """
    Validasi path input dan output
    
    Args:
        input_folder: path ke input folder
        output_folder: path ke output folder
        
    Returns:
        tuple: (input_path, output_path) as Path objects
        
    Raises:
        FileNotFoundError: jika input folder tidak ada
        ValueError: jika ada error pada path
    """
    input_path = Path(input_folder).resolve()
    output_path = Path(output_folder).resolve()
    
    # Validasi input folder
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder tidak ditemukan: {input_path}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path bukan folder: {input_path}")
    
    # Check apakah ada file .tif di input folder
    tif_files = list(input_path.glob('*.tif'))
    if not tif_files:
        raise ValueError(f"Tidak ada file .tif di input folder: {input_path}")
    
    print(f"✓ Input folder valid: {input_path}")
    print(f"  Total .tif files: {len(tif_files)}")
    
    return input_path, output_path


if __name__ == "__main__":
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validasi paths
        print("=" * 100)
        print("VALIDATION")
        print("=" * 100)
        input_path, output_path = validate_paths(args.input_folder, args.output_folder)
        
        # Print konfigurasi
        print(f"✓ Output folder akan dibuat/digunakan: {output_path}")
        print(f"\nConfiguration:")
        print(f"  Input folder:     {input_path}")
        print(f"  Output folder:    {output_path}")
        print(f"  Threshold:        {args.threshold}")
        print(f"  Persistence:      {args.persistence} bulan")
        print()
        
        # Process batch
        process_batch_deforestation(
            input_folder=str(input_path),
            output_folder=str(output_path),
            threshold=args.threshold,
            persistence_months=args.persistence
        )
        
        print("=" * 100)
        print("SUCCESS: Processing selesai!")
        print("=" * 100)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
    except NotADirectoryError as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Terjadi error tidak terduga: {e}")
        exit(1)