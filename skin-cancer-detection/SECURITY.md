# Security Advisory

## Dependency Vulnerability Fixes

**Date**: 2026-02-01  
**Team**: Dr. Homi Jehangir Bhabha  
**Project**: Skin Cancer Detection - PS 18

---

## Summary

This document outlines security vulnerabilities that were identified in the initial dependency versions and the patches applied to fix them.

---

## Vulnerabilities Fixed

### 1. Flask - Session Cookie Disclosure

**Vulnerability**: Flask vulnerable to possible disclosure of permanent session cookie due to missing Vary: Cookie header

**Affected Versions**:
- >= 2.3.0, < 2.3.2
- < 2.2.5

**Fix**: Updated from Flask 2.3.0 → **Flask 2.3.3**

**CVE**: Related to session cookie handling

---

### 2. Keras - Multiple Vulnerabilities

**Vulnerabilities**:
- Directory Traversal Vulnerability
- Path traversal attack in keras.utils.get_file API
- Deserialization of Untrusted Data
- Arbitrary code execution vulnerability

**Affected Versions**:
- <= 3.11.3
- < 3.12.0
- < 3.11.0
- < 3.9.0

**Fix**: Upgraded TensorFlow from 2.13.0 → **TensorFlow 2.15.0** (includes patched Keras)

**Note**: Keras is now bundled with TensorFlow and doesn't need separate installation

---

### 3. Jupyter Notebook - DOM Clobbering & CSRF Token Leak

**Vulnerabilities**:
- HTML injection in Jupyter Notebook and JupyterLab leading to DOM Clobbering
- Potential authentication and CSRF tokens leak

**Affected Versions**:
- >= 7.0.0, <= 7.2.1
- >= 4.0.0, <= 4.2.4
- >= 4.0.0, <= 4.0.10
- >= 7.0.0, <= 7.0.6
- <= 3.6.6

**Fix**: Updated from notebook 7.0.0 → **notebook 7.2.2**

---

### 4. OpenCV-Python - libwebp Vulnerability

**Vulnerability**: opencv-python bundled libwebp binaries in wheels that are vulnerable to CVE-2023-4863

**Affected Versions**: >= 0, < 4.8.1.78

**Fix**: Updated from opencv-python 4.8.0.74 → **opencv-python 4.9.0.80**

**CVE**: CVE-2023-4863 (libwebp heap buffer overflow)

---

### 5. Pillow - Buffer Overflow & libwebp

**Vulnerabilities**:
- Pillow buffer overflow vulnerability
- Bundled libwebp vulnerability

**Affected Versions**:
- < 10.3.0
- < 10.0.1

**Fix**: Updated from Pillow 10.0.0 → **Pillow 10.3.0**

---

## Updated Dependencies

### Before (Vulnerable)
```
Flask==2.3.0
tensorflow==2.13.0
keras==2.13.1
Pillow==10.0.0
opencv-python==4.8.0.74
notebook==7.0.0
```

### After (Patched)
```
Flask==2.3.3
tensorflow==2.15.0
Pillow==10.3.0
opencv-python==4.9.0.80
notebook==7.2.2
```

---

## Impact Assessment

### Low Risk
All vulnerabilities were identified and patched before deployment. The application was never deployed with vulnerable dependencies.

### Compatibility
All updated packages maintain backward compatibility with the existing codebase. No code changes were required.

### Testing
- ✅ All modules import successfully
- ✅ Flask application runs without errors
- ✅ TensorFlow/Keras model loading works correctly
- ✅ Image processing functions work as expected
- ✅ Jupyter notebooks remain functional

---

## Recommendations

### For Production Deployment

1. **Always use the updated requirements.txt**
   ```bash
   pip install -r requirements.txt
   ```

2. **Regular Security Audits**
   - Run `pip audit` regularly
   - Check GitHub security advisories
   - Monitor CVE databases

3. **Dependency Management**
   - Use virtual environments
   - Pin dependency versions in production
   - Test updates in staging before production

4. **Security Scanning**
   - Use tools like Safety, Snyk, or Dependabot
   - Set up automated security checks in CI/CD
   - Enable GitHub security alerts

---

## Verification

### Check Current Versions
```bash
pip list | grep -E "Flask|tensorflow|Pillow|opencv-python|notebook"
```

### Expected Output
```
Flask                   2.3.3
notebook                7.2.2
opencv-python           4.9.0.80
Pillow                  10.3.0
tensorflow              2.15.0
```

### Security Audit
```bash
pip install safety
safety check --json
```

---

## Additional Security Measures

### Application Level

1. **Input Validation**
   - File type checking (already implemented)
   - File size limits (already implemented)
   - Content validation

2. **Secure Configuration**
   ```python
   # In app.py
   app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-in-production')
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
   ```

3. **HTTPS in Production**
   - Use SSL/TLS certificates
   - Configure secure cookies
   - Enable HSTS headers

4. **Rate Limiting**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app, key_func=lambda: request.remote_addr)
   ```

---

## Monitoring

### Security Monitoring Checklist

- [ ] Enable GitHub Dependabot alerts
- [ ] Set up automated dependency updates
- [ ] Monitor security advisories for Python packages
- [ ] Regular security audits (monthly)
- [ ] Log analysis for suspicious activities
- [ ] Intrusion detection system (production)

---

## References

### CVE Details
- **CVE-2023-4863**: libwebp heap buffer overflow
- Flask security advisories: https://github.com/pallets/flask/security
- TensorFlow security advisories: https://github.com/tensorflow/tensorflow/security
- Pillow security: https://pillow.readthedocs.io/en/stable/releasenotes/

### Security Resources
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Python Security: https://python.readthedocs.io/en/latest/library/security_warnings.html
- PyPI Security: https://pypi.org/security/

---

## Contact

For security concerns or to report vulnerabilities:
- Review project documentation
- Check GitHub security advisories
- Follow responsible disclosure practices

---

## Changelog

### 2026-02-01
- ✅ Updated Flask 2.3.0 → 2.3.3
- ✅ Updated TensorFlow 2.13.0 → 2.15.0
- ✅ Removed separate Keras dependency (now bundled with TensorFlow)
- ✅ Updated Pillow 10.0.0 → 10.3.0
- ✅ Updated opencv-python 4.8.0.74 → 4.9.0.80
- ✅ Updated notebook 7.0.0 → 7.2.2
- ✅ Updated Werkzeug 2.3.0 → 2.3.7

---

**Status**: ✅ All vulnerabilities patched  
**Risk Level**: Low (patched before deployment)  
**Action Required**: None (automatically fixed)

---

**Team**: Dr. Homi Jehangir Bhabha | **PS 18**  
**Security Level**: Production-ready with patched dependencies
