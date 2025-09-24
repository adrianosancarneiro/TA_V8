# 🔒 TA_V8 RAG MCP Security Setup Guide

## Overview
The RAG_MCP project has been secured by removing all hardcoded credentials and implementing a centralized secrets management system.

## Security Features Implemented

### ✅ What Was Done
1. **Removed Hardcoded Credentials**: All passwords, API keys, and secrets removed from source code
2. **Secure Secrets File**: Created `/etc/TA_V8/RAG_MCP/secrets.env` with proper 600 permissions
3. **Configuration Module**: Added `shared/config.py` for centralized configuration management
4. **Security Validation**: Added `validate_security.py` script to scan for security issues
5. **Updated Docker Configuration**: Docker services now use the secrets file via `env_file`

### 🚫 What Was Removed
- Hardcoded passwords in `docker-compose.yml`
- Hardcoded database credentials in `tao_integration.py`
- Direct `os.getenv()` calls with hardcoded defaults

## 🚀 Deployment Steps

### 1. Update Secrets File
Edit the secure secrets file:
```bash
sudo nano /etc/TA_V8/RAG_MCP/secrets.env
```

**CRITICAL**: Replace ALL placeholder values:
```bash
# Example - REPLACE THESE VALUES
POSTGRES_PASSWORD=your_actual_strong_password_here
QDRANT_API_KEY=your_qdrant_api_key_if_needed
MCP_SECRET_KEY=$(openssl rand -hex 32)
API_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
```

### 2. Generate Strong Secrets
Use these commands to generate secure secrets:
```bash
# Generate random secrets
echo "MCP_SECRET_KEY=$(openssl rand -hex 32)"
echo "API_SECRET_KEY=$(openssl rand -hex 32)" 
echo "ENCRYPTION_KEY=$(openssl rand -hex 32)"
echo "MCP_JWT_SECRET=$(openssl rand -hex 32)"
```

### 3. Verify Security
Run the security validation:
```bash
cd /home/mentorius/AI_Services/TA_V8/RAG_MCP
python validate_security.py
```

### 4. Deploy with Secure Configuration
```bash
cd /home/mentorius/AI_Services/TA_V8/RAG_MCP
./deploy.sh
```

## 🔐 Security Best Practices

### File Permissions
```bash
# Secrets file should be 600 (owner read/write only)
chmod 600 /etc/TA_V8/RAG_MCP/secrets.env

# Verify permissions
ls -la /etc/TA_V8/RAG_MCP/secrets.env
# Should show: -rw------- 1 root root
```

### Environment Variables Priority
The system uses this priority order:
1. **Environment Variables** (highest priority)
2. **Secrets File** (`/etc/TA_V8/RAG_MCP/secrets.env`)
3. **Default Values** (lowest priority)

### Production Deployment
For production, consider:
```bash
# Use environment variables instead of secrets file
export POSTGRES_PASSWORD="your_production_password"
export API_SECRET_KEY="your_production_api_key"

# Or use Docker secrets
docker secret create postgres_password /path/to/password/file
```

## 🔍 Security Validation

### Run Security Checks
```bash
# Automated security scan
python validate_security.py

# Manual verification
grep -r "password.*=" . --exclude-dir=.venv
grep -r "key.*=" . --exclude-dir=.venv
```

### Expected Output
✅ **Secure Setup**:
- No hardcoded secrets in code
- Secrets file permissions: 600
- All placeholder values replaced

⚠️ **Needs Attention**:
- Placeholder values still present
- Insecure file permissions
- Hardcoded secrets found

## 📋 Configuration Reference

### Required Variables
```bash
# Database (Required)
POSTGRES_HOST=ta_v8_postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres_user
POSTGRES_PASSWORD=STRONG_PASSWORD_HERE
POSTGRES_DB=ta_v8

# Vector Database (Required)
QDRANT_HOST=ta_v8_qdrant
QDRANT_PORT=6333

# Services (Required)
EMBEDDING_URL=http://ta_v8_multilingual-e5-large:8080
OLLAMA_URL=http://ta_v8_ollama:11434
MCP_URL=http://rag-mcp-unified:8000
```

### Optional Variables
```bash
# Security
API_SECRET_KEY=generated_secret_key
MCP_SECRET_KEY=generated_secret_key
ENCRYPTION_KEY=generated_secret_key

# External APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

## 🚨 Emergency Procedures

### If Secrets Are Compromised
1. **Immediately rotate all secrets**:
   ```bash
   # Generate new secrets
   openssl rand -hex 32  # Use for new passwords
   ```

2. **Update secrets file**:
   ```bash
   sudo nano /etc/TA_V8/RAG_MCP/secrets.env
   ```

3. **Restart services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### If Secrets File Is Lost
1. **Recreate from template**:
   ```bash
   sudo cp /home/mentorius/AI_Services/TA_V8/RAG_MCP/secrets.env.template /etc/TA_V8/RAG_MCP/secrets.env
   sudo chmod 600 /etc/TA_V8/RAG_MCP/secrets.env
   ```

2. **Update with new values**
3. **Test configuration**

## ✅ Final Checklist

Before production deployment:
- [ ] All placeholder values replaced in secrets.env
- [ ] Secrets file permissions set to 600
- [ ] Security validation passes
- [ ] Strong, unique passwords generated
- [ ] Backup of secrets file created (stored securely)
- [ ] Team trained on secret management procedures
- [ ] Monitoring/alerting configured for security events

---

**Remember**: Security is an ongoing process. Regularly rotate secrets, audit access, and update this configuration as needed.