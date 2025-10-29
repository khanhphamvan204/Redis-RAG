#!/usr/bin/env python3
"""
Script để kiểm tra Redis và RediSearch module
Chạy: python redis_diagnostic.py
"""

import redis
import sys

def check_redis_connection():
    """Kiểm tra kết nối Redis"""
    print("=" * 60)
    print("KIỂM TRA KẾT NỐI REDIS")
    print("=" * 60)
    
    try:
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False
        )
        
        # Test ping
        response = client.ping()
        print(f"✅ Redis connection: OK (ping response: {response})")
        
        # Get Redis info
        info = client.info()
        print(f"✅ Redis version: {info.get('redis_version', 'unknown')}")
        print(f"✅ Redis mode: {info.get('redis_mode', 'unknown')}")
        
        return client
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("\n💡 Giải pháp:")
        print("   1. Kiểm tra Redis đang chạy: sudo systemctl status redis")
        print("   2. Khởi động Redis: sudo systemctl start redis")
        print("   3. Hoặc: redis-server")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def check_redis_modules(client):
    """Kiểm tra RediSearch module"""
    print("\n" + "=" * 60)
    print("KIỂM TRA REDISEARCH MODULE")
    print("=" * 60)
    
    try:
        modules = client.execute_command('MODULE LIST')
        
        if not modules:
            print("❌ Không có module nào được cài đặt")
            print_installation_guide()
            return False
        
        print(f"📦 Các module đã cài đặt: {len(modules)}")
        
        has_search = False
        for module in modules:
            # module is already a list of key-value pairs
            # Convert to dict for easier access
            module_dict = {}
            
            # Handle both dict and list formats
            if isinstance(module, dict):
                # Already a dict
                for key, value in module.items():
                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                    
                    if isinstance(value, bytes):
                        value = value.decode()
                    elif isinstance(value, list):
                        value = [v.decode() if isinstance(v, bytes) else v for v in value]
                    
                    module_dict[key_str] = value
            elif isinstance(module, list):
                # List of alternating keys and values
                for i in range(0, len(module), 2):
                    if i + 1 < len(module):
                        key = module[i]
                        value = module[i + 1]
                        
                        key_str = key.decode() if isinstance(key, bytes) else str(key)
                        
                        if isinstance(value, bytes):
                            value = value.decode()
                        elif isinstance(value, list):
                            value = [v.decode() if isinstance(v, bytes) else v for v in value]
                        
                        module_dict[key_str] = value
            
            module_name = module_dict.get('name', 'unknown')
            module_ver = module_dict.get('ver', 'unknown')
            
            print(f"   - {module_name} (version: {module_ver})")
            
            # Check if this is RediSearch module
            if isinstance(module_name, str) and 'search' in module_name.lower():
                has_search = True
        
        if has_search:
            print("\n✅ RediSearch module: INSTALLED")
            return True
        else:
            print("\n❌ RediSearch module: NOT FOUND")
            print_installation_guide()
            return False
            
    except redis.exceptions.ResponseError as e:
        print(f"❌ Không thể kiểm tra modules: {e}")
        print("💡 Redis server có thể không hỗ trợ MODULE LIST command")
        print_installation_guide()
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_installation_guide():
    """In hướng dẫn cài đặt"""
    print("\n" + "=" * 60)
    print("HƯỚNG DẪN CÀI ĐẶT REDIS STACK")
    print("=" * 60)
    
    print("\n🐧 Ubuntu/Debian:")
    print("   curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg")
    print("   echo 'deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main' | sudo tee /etc/apt/sources.list.d/redis.list")
    print("   sudo apt-get update")
    print("   sudo apt-get install redis-stack-server")
    print("   sudo systemctl start redis-stack-server")
    
    print("\n🍎 MacOS:")
    print("   brew tap redis-stack/redis-stack")
    print("   brew install redis-stack")
    print("   redis-stack-server")
    
    print("\n🐳 Docker:")
    print("   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest")
    
    print("\n📚 Hoặc:")
    print("   Website: https://redis.io/docs/stack/")
    print("   Download: https://redis.io/download")

def check_existing_indices(client):
    """Kiểm tra các index đã tồn tại"""
    print("\n" + "=" * 60)
    print("KIỂM TRA CÁC INDEX ĐÃ TẠO")
    print("=" * 60)
    
    try:
        # Try to list all indices
        keys = client.keys("doc:*")
        
        if keys:
            print(f"📊 Tìm thấy {len(keys)} keys với pattern 'doc:*'")
            
            # Group by file_type
            file_types = set()
            for key in keys[:10]:  # Show first 10
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(':')
                if len(parts) >= 2:
                    file_types.add(parts[1])
                print(f"   - {key_str}")
            
            if len(keys) > 10:
                print(f"   ... và {len(keys) - 10} keys khác")
            
            if file_types:
                print(f"\n📁 File types được tìm thấy: {', '.join(file_types)}")
        else:
            print("⚠️  Chưa có dữ liệu vector nào trong Redis")
            print("💡 Hãy thêm tài liệu qua API /add endpoint trước")
        
        # Try to get index info
        try:
            result = client.execute_command('FT._LIST')
            if result:
                print(f"\n📑 RediSearch indices: {len(result)}")
                for idx in result:
                    idx_name = idx.decode() if isinstance(idx, bytes) else idx
                    print(f"   - {idx_name}")
                    
                    try:
                        info = client.execute_command('FT.INFO', idx_name)
                        # Parse FT.INFO output (it's a list of key-value pairs)
                        for i in range(0, len(info), 2):
                            if i + 1 < len(info):
                                key = info[i].decode() if isinstance(info[i], bytes) else info[i]
                                if key == 'num_docs':
                                    num_docs = info[i + 1]
                                    print(f"     → Số documents: {num_docs}")
                                    break
                    except Exception as e:
                        print(f"     → Không thể lấy info: {e}")
            else:
                print("\n⚠️  Chưa có index nào được tạo")
        except redis.exceptions.ResponseError:
            print("\n⚠️  Không thể lấy danh sách indices (RediSearch chưa được cài)")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n🔍 BẮT ĐẦU KIỂM TRA HỆ THỐNG REDIS\n")
    
    # Step 1: Check connection
    client = check_redis_connection()
    if not client:
        sys.exit(1)
    
    # Step 2: Check modules
    has_search = check_redis_modules(client)
    
    # Step 3: Check existing data
    check_existing_indices(client)
    
    # Summary
    print("\n" + "=" * 60)
    print("TÓM TẮT")
    print("=" * 60)
    
    if has_search:
        print("✅ Hệ thống sẵn sàng sử dụng RedisVL")
        print("💡 Bạn có thể thêm tài liệu qua API /add endpoint")
    else:
        print("❌ Hệ thống CHƯA sẵn sàng")
        print("💡 Vui lòng cài đặt Redis Stack theo hướng dẫn ở trên")
        print("\n🔄 HOẶC: Quay lại sử dụng FAISS (không cần Redis)")
    
    print("\n")

if __name__ == "__main__":
    main()