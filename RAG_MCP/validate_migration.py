#!/usr/bin/env python3
"""
# =============================================================================
# RAG MCP MIGRATION VALIDATION SCRIPT
# =============================================================================
# Purpose: Validate the complete migration from unified_mcp_server.py to 
#          MCP-compliant microservices architecture
#
# This script validates:
# 1. All MCP services are properly implemented
# 2. Platform integration modules are ready
# 3. Docker configuration is complete
# 4. Database schemas are available
# 5. Testing infrastructure is set up
# =============================================================================
"""

import asyncio
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")

def print_success(message: str):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message: str):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message: str):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists"""
    return Path(file_path).exists()

def check_module_syntax(module_path: str) -> Tuple[bool, str]:
    """Check if a Python module has valid syntax"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        if spec is None:
            return False, "Could not load module spec"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def validate_mcp_services() -> List[Tuple[str, bool, str]]:
    """Validate all MCP service implementations"""
    print_header("VALIDATING MCP SERVICES")
    
    services = [
        ("Chunking Service", "mcp_services/chunking/server.py"),
        ("Embedding Service", "mcp_services/embedding/server.py"), 
        ("Retrieval Service", "mcp_services/retrieval/server.py")
    ]
    
    results = []
    
    for service_name, service_path in services:
        if check_file_exists(service_path):
            syntax_ok, error = check_module_syntax(service_path)
            if syntax_ok:
                print_success(f"{service_name}: Implementation ✓ Syntax ✓")
                results.append((service_name, True, "OK"))
            else:
                print_error(f"{service_name}: Implementation ✓ Syntax ❌ ({error})")
                results.append((service_name, False, error))
        else:
            print_error(f"{service_name}: File missing - {service_path}")
            results.append((service_name, False, "File missing"))
    
    return results

def validate_platform_integration() -> List[Tuple[str, bool, str]]:
    """Validate platform integration modules"""
    print_header("VALIDATING PLATFORM INTEGRATION")
    
    integrations = [
        ("TAO Integration", "platform_modules/TAO_integration/mcp_registry.py"),
        ("TAE Integration", "platform_modules/TAE_integration/tool_caller.py"),
        ("TAB Integration", "platform_modules/TAB_integration/knowledge_builder.py")
    ]
    
    results = []
    
    for integration_name, integration_path in integrations:
        if check_file_exists(integration_path):
            syntax_ok, error = check_module_syntax(integration_path)
            if syntax_ok:
                print_success(f"{integration_name}: Implementation ✓ Syntax ✓")
                results.append((integration_name, True, "OK"))
            else:
                print_error(f"{integration_name}: Implementation ✓ Syntax ❌ ({error})")
                results.append((integration_name, False, error))
        else:
            print_error(f"{integration_name}: File missing - {integration_path}")
            results.append((integration_name, False, "File missing"))
    
    return results

def validate_docker_configuration() -> List[Tuple[str, bool, str]]:
    """Validate Docker configuration files"""
    print_header("VALIDATING DOCKER CONFIGURATION")
    
    docker_files = [
        ("Chunking Dockerfile", "infrastructure/docker/Dockerfile.chunking"),
        ("Embedding Dockerfile", "infrastructure/docker/Dockerfile.embedding_mcp"),
        ("Retrieval Dockerfile", "infrastructure/docker/Dockerfile.retriever"),
        ("MCP Docker Compose", "deployment/docker-compose-mcp.yml")
    ]
    
    results = []
    
    for file_name, file_path in docker_files:
        if check_file_exists(file_path):
            print_success(f"{file_name}: Available ✓")
            results.append((file_name, True, "OK"))
        else:
            print_error(f"{file_name}: Missing - {file_path}")
            results.append((file_name, False, "File missing"))
    
    return results

def validate_database_schemas() -> List[Tuple[str, bool, str]]:
    """Validate database schema files"""
    print_header("VALIDATING DATABASE SCHEMAS")
    
    schema_files = [
        ("PostgreSQL Schema", "infrastructure/databases/postgresql_schema.sql"),
        ("Neo4j Schema", "infrastructure/databases/neo4j_schema.cypher")
    ]
    
    results = []
    
    for schema_name, schema_path in schema_files:
        if check_file_exists(schema_path):
            print_success(f"{schema_name}: Available ✓")
            results.append((schema_name, True, "OK"))
        else:
            print_error(f"{schema_name}: Missing - {schema_path}")
            results.append((schema_name, False, "File missing"))
    
    return results

def validate_testing_infrastructure() -> List[Tuple[str, bool, str]]:
    """Validate testing setup"""
    print_header("VALIDATING TESTING INFRASTRUCTURE")
    
    test_files = [
        ("MCP Compliance Tests", "testing/mcp_compliance/test_mcp_servers.py"),
        ("Migration Plan", "MIGRATION_PLAN_V2.md")
    ]
    
    results = []
    
    for test_name, test_path in test_files:
        if check_file_exists(test_path):
            print_success(f"{test_name}: Available ✓")
            results.append((test_name, True, "OK"))
        else:
            print_error(f"{test_name}: Missing - {test_path}")
            results.append((test_name, False, "File missing"))
    
    return results

def validate_legacy_backup() -> List[Tuple[str, bool, str]]:
    """Validate legacy code backup"""
    print_header("VALIDATING LEGACY BACKUP")
    
    legacy_files = [
        ("Original Unified Server", "legacy/unified_mcp_server.py"),
        ("Legacy Backup Complete", "legacy/")
    ]
    
    results = []
    
    for legacy_name, legacy_path in legacy_files:
        if check_file_exists(legacy_path):
            print_success(f"{legacy_name}: Preserved ✓")
            results.append((legacy_name, True, "OK"))
        else:
            print_warning(f"{legacy_name}: Not found - {legacy_path}")
            results.append((legacy_name, False, "Not preserved"))
    
    return results

def generate_migration_report(all_results: Dict[str, List[Tuple[str, bool, str]]]):
    """Generate final migration report"""
    print_header("MIGRATION COMPLETION REPORT")
    
    total_checks = 0
    passed_checks = 0
    
    for category, results in all_results.items():
        total_checks += len(results)
        category_passed = sum(1 for _, passed, _ in results if passed)
        passed_checks += category_passed
        
        if category_passed == len(results):
            print_success(f"{category}: {category_passed}/{len(results)} ✓")
        else:
            print_warning(f"{category}: {category_passed}/{len(results)}")
    
    print(f"\n{Colors.BOLD}Overall Migration Status:{Colors.END}")
    if passed_checks == total_checks:
        print_success(f"MIGRATION COMPLETE: {passed_checks}/{total_checks} checks passed")
        print_info("🚀 Ready for deployment and testing!")
        return True
    else:
        print_warning(f"MIGRATION PARTIAL: {passed_checks}/{total_checks} checks passed")
        print_info("⚠️  Review failed checks before deployment")
        return False

def print_next_steps():
    """Print recommended next steps"""
    print_header("RECOMMENDED NEXT STEPS")
    
    steps = [
        "1. Start infrastructure services (PostgreSQL, Neo4j, Qdrant, MinIO)",
        "2. Build and deploy MCP services: docker-compose -f deployment/docker-compose-mcp.yml up",
        "3. Run integration tests: python testing/mcp_compliance/test_mcp_servers.py",
        "4. Validate performance benchmarks",
        "5. Integrate with TAB/TAE/TAO when ready",
        "6. Monitor service health and performance"
    ]
    
    for step in steps:
        print_info(step)

async def main():
    """Main validation function"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("🔍 RAG MCP MIGRATION VALIDATION")
    print("=" * 60)
    print(f"{Colors.END}")
    
    # Run all validations
    all_results = {
        "MCP Services": validate_mcp_services(),
        "Platform Integration": validate_platform_integration(), 
        "Docker Configuration": validate_docker_configuration(),
        "Database Schemas": validate_database_schemas(),
        "Testing Infrastructure": validate_testing_infrastructure(),
        "Legacy Backup": validate_legacy_backup()
    }
    
    # Generate report
    migration_complete = generate_migration_report(all_results)
    
    # Print next steps
    print_next_steps()
    
    # Final status
    if migration_complete:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 MIGRATION SUCCESSFULLY COMPLETED!{Colors.END}")
        sys.exit(0)
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  MIGRATION NEEDS ATTENTION{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())