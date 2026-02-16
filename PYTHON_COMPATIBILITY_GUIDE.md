# Python Compatibility Prevention Guide

## 🚨 What Happened
- **Problem**: Python 3.14 removed `formatargspec` from `inspect` module
- **Impact**: `wrapt` package (dependency of llama-index) failed to install
- **Solution**: Used Python 3.13.4 with compatible package versions

## 🛡️ Prevention Strategies

### 1. **Use Stable Python Versions**
```bash
# ✅ RECOMMENDED: Use stable versions
python3.13 -m venv venv  # Python 3.13.x (current stable)
python3.12 -m venv venv  # Python 3.12.x (also stable)
python3.11 -m venv venv  # Python 3.11.x (LTS)

# ❌ AVOID: Bleeding edge versions
python3.14 -m venv venv  # Too new, packages not updated yet
```

### 2. **Pin Package Versions**
Your current `requirements.txt` uses unpinned versions, which can cause issues:

**Current (problematic):**
```
Flask
llama-index
openai
```

**Better (pinned):**
```
Flask==3.1.2
llama-index==0.14.6
openai==1.109.1
```

**Use the pinned version:**
```bash
# Copy the working versions
cp requirements_pinned.txt requirements.txt
```

### 3. **Test Before Upgrading**
```bash
# Before upgrading Python or packages:
python --version  # Check current version
pip list          # Check current packages
pip freeze > backup_requirements.txt  # Backup working versions
```

### 4. **Use Virtual Environments Always**
```bash
# Create project-specific environments
python3.13 -m venv venv_project_name
source venv_project_name/bin/activate

# Never install packages globally
pip install --user package_name  # Still risky
pip install package_name         # Dangerous
```

### 5. **Check Package Compatibility**
```bash
# Check if package supports your Python version
pip show package_name
python -c "import package_name; print('OK')"
```

### 6. **Use Compatibility Tools**
```bash
# Install pip-tools for better dependency management
pip install pip-tools

# Create requirements.in with loose constraints
echo "Flask>=3.0" > requirements.in
echo "llama-index>=0.14" >> requirements.in

# Generate locked requirements.txt
pip-compile requirements.in
```

### 7. **Monitor Python Release Schedule**
- **Python 3.13**: Current stable (recommended)
- **Python 3.14**: Very new (avoid for production)
- **Python 3.15**: Future release (avoid)

Check: https://www.python.org/downloads/

## 🔧 Quick Fixes for Common Issues

### Issue: `formatargspec` not found
```bash
# Solution: Use Python 3.13 or earlier
python3.13 -m venv venv_new
source venv_new/bin/activate
pip install -r requirements.txt
```

### Issue: Package version conflicts
```bash
# Solution: Use pinned versions
pip freeze > requirements_pinned.txt
pip install -r requirements_pinned.txt
```

### Issue: Build failures
```bash
# Solution: Install build dependencies first
pip install --upgrade pip setuptools wheel
pip install package_name
```

## 📋 Best Practices Checklist

- [ ] Use Python 3.11, 3.12, or 3.13 (avoid 3.14+)
- [ ] Pin all package versions in requirements.txt
- [ ] Use virtual environments for every project
- [ ] Test installations in clean environments
- [ ] Keep backup of working requirements.txt
- [ ] Check package compatibility before upgrading
- [ ] Use `pip-tools` for dependency management
- [ ] Monitor Python release announcements

## 🚀 Your Current Working Setup

**Environment**: `venv_python313/`
**Python**: 3.13.4
**Status**: ✅ All packages working

**To use:**
```bash
cd /Users/jorgefernandezilufi/Documents/_article_research/rag-system
source venv_python313/bin/activate
```

**To recreate if needed:**
```bash
python3.13 -m venv venv_python313
source venv_python313/bin/activate
pip install -r requirements_pinned.txt
```

## 📞 Emergency Recovery

If you encounter similar issues:

1. **Identify the error** (like `formatargspec` not found)
2. **Check Python version** (`python --version`)
3. **Downgrade Python** if using bleeding edge version
4. **Use pinned requirements** (`requirements_pinned.txt`)
5. **Create fresh environment** with stable Python version

Remember: **Stability over bleeding edge features!**







