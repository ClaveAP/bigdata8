"""
Modul Inisialisasi Apache Spark Session
Dikonfigurasi optimal untuk pemrosesan skala 1.000.000 baris data secara lokal (multi-core).
"""

import os
from pyspark.sql import SparkSession

def get_spark_session(app_name: str = "EcommerceAbandonmentAnalysis", driver_memory: str = "4g") -> SparkSession:
    """
    Membuat atau mengembalikan SparkSession yang telah dikonfigurasi.
    
    Args:
        app_name (str): Nama aplikasi Spark.
        driver_memory (str): Alokasi memori heap untuk driver (default: 4GB).
        
    Returns:
        SparkSession: Objek SparkSession aktif.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # Menggunakan seluruh core CPU yang tersedia di mesin lokal
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # Optimasi interop Apache Arrow
        .config("spark.sql.shuffle.partitions", "8")  # Mengurangi overhead shuffle di mode lokal
        .getOrCreate()
    )
    
    # Set level log agar tidak terlalu verbose di terminal/notebook
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

if __name__ == "__main__":
    print("[INFO] Menguji inisialisasi PySpark Session...")
    spark = get_spark_session()
    print(f"[SUCCESS] SparkSession aktif! Versi Spark: {spark.version}")
    spark.stop()
