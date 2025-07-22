#!/usr/bin/env python3
"""
GGUF Model Sharding Tool

Converts standard GGUF model files into memory-efficient sharded format
with hardware-accelerated CRC32 integrity validation.

Usage: python model_sharding_tool.py shard <input.gguf>
"""

import os
import sys
import json
import struct
import zlib  # Use zlib for CRC32 instead of hashlib
from pathlib import Path
from typing import List, Dict, Any

class GGUFShard:
    MAGIC = b'SGUF'
    VERSION = 1
    PAGE_SIZE = 4096
    
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.output_dir = self.input_file.parent
        self.core_file = self.output_dir / "core.gguf"
        self.map_file = self.output_dir / "core.sgmap"
        
    def create_shards(self) -> bool:
        """Create sharded GGUF files from input"""
        try:
            print(f"Forging shards from {self.input_file}")
            
            # Remove existing files if they exist
            if os.path.exists(self.core_file):
                print(f"Removing existing {self.core_file}")
                os.remove(self.core_file)
                
            if os.path.exists(self.map_file):
                print(f"Removing existing {self.map_file}")
                os.remove(self.map_file)
            
            # Read input file
            with open(self.input_file, 'rb') as f:
                data = f.read()
            
            file_size = len(data)
            page_count = (file_size + self.PAGE_SIZE - 1) // self.PAGE_SIZE
            
            print(f"File size: {file_size:,} bytes")
            print(f"Pages: {page_count}")
            
            # Create shard map
            shard_map = self._create_shard_map(data, page_count)
            
            # Write core file with sharded data
            self._write_core_file(data, shard_map)
            
            # Write shard map
            self._write_shard_map(shard_map)
            
            print(f"Created {self.core_file}")
            print(f"Created {self.map_file}")
            
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def _create_shard_map(self, data: bytes, page_count: int) -> Dict[str, Any]:
        """Create shard mapping metadata"""
        shards = []
        
        for i in range(page_count):
            offset = i * self.PAGE_SIZE
            page_data = data[offset:offset + self.PAGE_SIZE]
            
            # For shard map, we need to calculate CRC for the actual content that will be stored
            # This should be the page data truncated to PAGE_SIZE - 8 bytes
            if len(page_data) < self.PAGE_SIZE - 8:
                # Pad to PAGE_SIZE - 8 bytes
                content_for_crc = page_data + b'\x00' * (self.PAGE_SIZE - 8 - len(page_data))
            else:
                # Truncate to PAGE_SIZE - 8 bytes
                content_for_crc = page_data[:self.PAGE_SIZE - 8]
            
            # Calculate CRC32 for the content that will be stored
            crc32 = zlib.crc32(content_for_crc) & 0xffffffff
            
            shard = {
                "id": i,
                "offset": offset,
                "size": self.PAGE_SIZE,  # Full page size including CRC tag
                "crc32": f"0x{crc32:08x}",
                "file": "core.gguf",
                "priority": "high" if i < 16 else "normal"  # First 16 pages high priority
            }
            shards.append(shard)
        
        return {
            "version": "1.0",
            "source_file": str(self.input_file.name),
            "total_shards": page_count,
            "page_size": self.PAGE_SIZE,
            "shards": shards,
            "atlas": {
                "memory_layout": "column_major",
                "cache_policy": "lru", 
                "prefetch_distance": 8
            }
        }
    
    def _write_core_file(self, data: bytes, shard_map: Dict[str, Any]) -> None:
        """Write core sharded file"""
        with open(self.core_file, 'wb') as f:
            # Write SGUF header (Sharded GGUF format) - exactly 256 bytes
            header = bytearray(256)
            
            # Pack the header components
            struct.pack_into('<4sII', header, 0, self.MAGIC, self.VERSION, len(shard_map['shards']))
            struct.pack_into('<QQ', header, 12, self.PAGE_SIZE, len(data))
            
            # Calculate header CRC for the first 252 bytes (leave last 4 bytes for CRC)
            header_crc = zlib.crc32(header[:252]) & 0xffffffff
            struct.pack_into('<I', header, 252, header_crc)
            
            f.write(header)
            
            # Write pages with CRC tags
            for i in range(len(shard_map['shards'])):
                offset = i * self.PAGE_SIZE
                page_data = data[offset:offset + self.PAGE_SIZE]
                
                # Ensure page data is exactly PAGE_SIZE - 8 bytes (reserve 8 bytes for CRC tag)
                if len(page_data) < self.PAGE_SIZE - 8:
                    page_data += b'\x00' * (self.PAGE_SIZE - 8 - len(page_data))
                elif len(page_data) > self.PAGE_SIZE - 8:
                    page_data = page_data[:self.PAGE_SIZE - 8]
                
                # Calculate CRC32 for the page content
                crc32 = zlib.crc32(page_data) & 0xffffffff
                
                # Create CRC tag: 4 bytes CRC32 + 'PGCR' magic
                crc_tag = struct.pack('<I', crc32) + b'PGCR'
                
                # Combine page data with CRC tag
                full_page = page_data + crc_tag
                
                # Verify the page is exactly PAGE_SIZE
                assert len(full_page) == self.PAGE_SIZE, f"Page {i} has wrong size: {len(full_page)}"
                
                f.write(full_page)
    
    def _write_shard_map(self, shard_map: Dict[str, Any]) -> None:
        """Write shard map JSON file"""
        with open(self.map_file, 'w') as f:
            json.dump(shard_map, f, indent=2)

def main():
    if len(sys.argv) < 3 or sys.argv[1] != 'shard':
        print("Usage: python model_sharding_tool.py shard <input.gguf>")
        sys.exit(1)
    
    input_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    forge = GGUFShard(input_file)
    success = forge.create_shards()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
