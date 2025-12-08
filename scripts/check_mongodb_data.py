"""
Quick script to check MongoDB collections for Spark analytics data
"""
from pymongo import MongoClient
import json
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/faiss_db")
db = client["faiss_db"]

print("=" * 60)
print("CHECKING MONGODB COLLECTIONS FOR ANALYTICS DATA")
print("=" * 60)

# Expected collections from Spark
spark_collections = [
    "query_analytics_by_faculty",
    "query_analytics_by_year", 
    "query_analytics_heatmap"
]

# Also check raw query logs
all_collections = ["query_logs"] + spark_collections

for coll_name in all_collections:
    print(f"\n📊 Collection: {coll_name}")
    print("-" * 60)
    
    try:
        collection = db[coll_name]
        count = collection.count_documents({})
        
        if count == 0:
            print(f"   ❌ Empty (0 documents)")
        else:
            print(f"   ✅ Found {count} documents")
            
            # Show sample document
            sample = collection.find_one()
            if sample:
                # Remove _id for cleaner display
                if '_id' in sample:
                    del sample['_id']
                print(f"\n   Sample document:")
                print(f"   {json.dumps(sample, indent=2, default=str)}")
                
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Check if pipeline is working
query_logs_count = db["query_logs"].count_documents({})
faculty_count = db["query_analytics_by_faculty"].count_documents({})
year_count = db["query_analytics_by_year"].count_documents({})
heatmap_count = db["query_analytics_heatmap"].count_documents({})

print(f"\n📈 Data Pipeline Status:")
print(f"   Raw Queries (MongoDB): {query_logs_count}")
print(f"   Faculty Analytics (Spark): {faculty_count}")
print(f"   Year Analytics (Spark): {year_count}")
print(f"   Heatmap Data (Spark): {heatmap_count}")

if query_logs_count > 0:
    print(f"\n✅ Query tracking is working!")
    if faculty_count == 0 and year_count == 0:
        print(f"⚠️  But Spark aggregations not found yet.")
        print(f"   → Check if Spark Streaming container is running")
        print(f"   → Check Spark logs: docker logs spark-streaming")
else:
    print(f"\n❌ No query logs found!")
    print(f"   → Make some queries in the chat UI to generate data")

client.close()
