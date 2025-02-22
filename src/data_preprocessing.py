import pandas as pd
import gzip

# File paths
input_vcf_file = "data/gnomad.exomes.v4.1.sites.chr21.vcf.bgz"
output_csv_file = "outputs/processed_vcf_data.csv"

def load_vcf(file_path):
    """Load VCF file and extract relevant columns."""
    print("🚀 Loading VCF data...")
    records = []
    
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip header lines
            parts = line.strip().split("\t")
            
            # Extract important columns
            chrom, pos, ref, alt = parts[0], int(parts[1]), parts[3], parts[4]
            
            # Example: Extracting more numeric fields
            qual = float(parts[5]) if parts[5] != "." else 0  # Convert QUAL to numeric
            dp = int(parts[7].split("DP=")[1].split(";")[0]) if "DP=" in parts[7] else 0  # Extract DP value
            
            records.append([chrom, pos, ref, alt, qual, dp])
    
    df = pd.DataFrame(records, columns=["Chrom", "Position", "Ref", "Alt", "Quality", "Depth"])
    
    print(f"✅ Loaded {len(df)} records.")
    
    return df

def preprocess_data(df):
    """Preprocess the data: Encode categorical variables and keep numeric features."""
    print("\n🔍 Dropping non-numeric columns...")
    df.drop(columns=["Chrom", "Ref"], inplace=True)  # Keep only numeric features

    print("✅ Non-numeric columns dropped.")

    # Encode 'Alt' column with unique values
    print("\n🔍 Encoding 'Alt' values...")
    df['Alt_encoded'] = df['Alt'].astype('category').cat.codes
    df.drop(columns=["Alt"], inplace=True)

    print("✅ 'Alt' column encoded.")

    return df

# Load and preprocess data
df = load_vcf(input_vcf_file)
df = preprocess_data(df)

# Save processed data
df.to_csv(output_csv_file, index=False)
print(f"✅ Processed data saved to {output_csv_file}")

# Display summary
print("\n🔍 Processed Data Sample:")
print(df.head())

print("\n🔍 Final Processed Features:")
print(df.columns.tolist())
