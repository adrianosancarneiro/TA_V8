#!/usr/bin/env python3
"""
Security validation script for TA_V8 RAG MCP
Checks for hardcoded credentials and validates security configuration
"""

import os
import re
from pathlib import Path
import sys

def scan_for_hardcoded_secrets(directory: Path) -> list[tuple[str, str, str]]:
    """
    Scan directory for hardcoded secrets
    Returns list of (file_path, line_number, matched_line)
    """
    secrets_patterns = [
        r'password\s*=\s*["\'][^"\']*["\']',
        r'key\s*=\s*["\'][^"\']*["\']',
        r'secret\s*=\s*["\'][^"\']*["\']',
        r'token\s*=\s*["\'][^"\']*["\']',
        r'POSTGRES_PASSWORD\s*=\s*[^$][^\s]*',
        r'API_KEY\s*=\s*[^$][^\s]*',
        r'SECRET_KEY\s*=\s*[^$][^\s]*',
    ]
    
    issues = []
    
    # Files to scan
    scan_files = [
        '*.py', '*.yml', '*.yaml', '*.json', '*.env', '*.sh',
        '*.md', '*.txt', '*.toml'
    ]
    
    # Exclude patterns
    exclude_patterns = [
        '.venv/', '__pycache__/', '.git/', 'node_modules/',
        'your_', 'example_', 'placeholder_', 'secrets.env'
    ]
    
    for pattern in scan_files:
        for file_path in directory.rglob(pattern):
            # Skip excluded paths
            if any(exclude in str(file_path) for exclude in exclude_patterns):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        for secret_pattern in secrets_patterns:
                            if re.search(secret_pattern, line, re.IGNORECASE):
                                # Skip if it's clearly a placeholder or example
                                if any(placeholder in line.lower() for placeholder in 
                                      ['your_', 'example_', 'placeholder_', 'xxx', '***']):
                                    continue
                                issues.append((str(file_path), line_num, line.strip()))
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
    
    return issues

def check_secrets_file_permissions(secrets_path: Path) -> bool:
    """Check if secrets file has correct permissions (600)"""
    if not secrets_path.exists():
        return False
    
    try:
        stat = secrets_path.stat()
        # Check if permissions are 600 (owner read/write only)
        permissions = oct(stat.st_mode)[-3:]
        return permissions == '600'
    except Exception:
        return False

def validate_secrets_file(secrets_path: Path) -> list[str]:
    """Validate secrets file configuration"""
    issues = []
    
    if not secrets_path.exists():
        issues.append("Secrets file does not exist")
        return issues
    
    required_vars = [
        'POSTGRES_PASSWORD', 'POSTGRES_USER', 'POSTGRES_HOST',
        'QDRANT_HOST', 'EMBEDDING_URL', 'OLLAMA_URL'
    ]
    
    try:
        with open(secrets_path, 'r') as f:
            content = f.read()
            
            for var in required_vars:
                if f"{var}=" not in content:
                    issues.append(f"Missing required variable: {var}")
                elif f"{var}=your_" in content:
                    issues.append(f"Placeholder value for: {var}")
                    
    except Exception as e:
        issues.append(f"Could not read secrets file: {e}")
    
    return issues

def main():
    """Main security validation"""
    print("🔒 TA_V8 RAG MCP Security Validation")
    print("=" * 50)
    
    project_dir = Path(__file__).parent
    secrets_file = Path("/etc/TA_V8/RAG_MCP/secrets.env")
    
    # Check for hardcoded secrets
    print("🔍 Scanning for hardcoded secrets...")
    hardcoded_issues = scan_for_hardcoded_secrets(project_dir)
    
    if hardcoded_issues:
        print(f"⚠️  Found {len(hardcoded_issues)} potential hardcoded secrets:")
        for file_path, line_num, line in hardcoded_issues:
            print(f"  📄 {file_path}:{line_num} - {line[:80]}...")
    else:
        print("✅ No hardcoded secrets found")
    
    # Check secrets file permissions
    print("\n🔐 Checking secrets file permissions...")
    if check_secrets_file_permissions(secrets_file):
        print("✅ Secrets file permissions are secure (600)")
    else:
        print("⚠️  Secrets file permissions may be insecure")
        print("   Run: chmod 600 /etc/TA_V8/RAG_MCP/secrets.env")
    
    # Validate secrets file
    print("\n📋 Validating secrets file configuration...")
    secrets_issues = validate_secrets_file(secrets_file)
    
    if secrets_issues:
        print(f"⚠️  Found {len(secrets_issues)} configuration issues:")
        for issue in secrets_issues:
            print(f"  • {issue}")
    else:
        print("✅ Secrets file configuration is valid")
    
    # Security recommendations
    print("\n💡 Security Recommendations:")
    print("  • Keep secrets.env file permissions at 600")
    print("  • Use strong, unique passwords for all services")
    print("  • Rotate secrets regularly")
    print("  • Never commit secrets to version control")
    print("  • Use environment variables in production")
    print("  • Enable firewall for production deployments")
    
    # Summary
    total_issues = len(hardcoded_issues) + len(secrets_issues)
    if not check_secrets_file_permissions(secrets_file):
        total_issues += 1
    
    print(f"\n📊 Security Summary: {total_issues} issues found")
    
    if total_issues == 0:
        print("🎉 All security checks passed!")
        return 0
    else:
        print("🚨 Please address the security issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())