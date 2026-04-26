from datasets import load_dataset

print("=== jordankzf/linkedin ===")
try:
    ds = load_dataset("jordankzf/linkedin", split="train[:5]")
    print("columns:", ds.column_names)
    for row in ds:
        print(row)
        print("---")
except Exception as e:
    print(f"실패: {e}")

print("\n=== cverse/linkedin ===")
try:
    ds2 = load_dataset("cverse/linkedin", split="train[:5]")
    print("columns:", ds2.column_names)
    for row in ds2:
        print(row)
        print("---")
except Exception as e:
    print(f"실패: {e}")

print("\n=== NeuML/neuml-linkedin-202501 ===")
try:
    ds3 = load_dataset("NeuML/neuml-linkedin-202501", split="train[:5]")
    print("columns:", ds3.column_names)
    for row in ds3:
        print(row)
        print("---")
except Exception as e:
    print(f"실패: {e}")
