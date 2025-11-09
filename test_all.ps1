# Quick Test Script
# Tests all phases of the spam classifier

Write-Host "=" * 70
Write-Host "Spam Classifier - Quick Test Suite" -ForegroundColor Cyan
Write-Host "=" * 70

# Activate virtual environment
& .venv\Scripts\Activate.ps1

Write-Host "`nTest 1: Phase 1 Baseline Model" -ForegroundColor Yellow
Write-Host "-" * 70
python scripts\predict_spam.py `
    --model models\spam_classifier.pkl `
    --vectorizer models\tfidf_vectorizer.pkl `
    --text "Win a FREE prize! Call 555-1234 NOW!"

Write-Host "`n`nTest 2: Phase 2 High Recall Model" -ForegroundColor Yellow
Write-Host "-" * 70
python scripts\predict_spam.py `
    --model models\spam_classifier_phase2.pkl `
    --vectorizer models\tfidf_vectorizer_phase2.pkl `
    --text "Win a FREE prize! Call 555-1234 NOW!"

Write-Host "`n`nTest 3: Phase 3 Balanced Model (RECOMMENDED)" -ForegroundColor Yellow
Write-Host "-" * 70
python scripts\predict_spam.py `
    --model models\spam_classifier_phase3.pkl `
    --vectorizer models\tfidf_vectorizer_phase3.pkl `
    --text "Win a FREE prize! Call 555-1234 NOW!"

Write-Host "`n`nTest 4: Ham Message Test" -ForegroundColor Yellow
Write-Host "-" * 70
python scripts\predict_spam.py `
    --model models\spam_classifier_phase3.pkl `
    --vectorizer models\tfidf_vectorizer_phase3.pkl `
    --text "Hey, are you free for coffee tomorrow afternoon?"

Write-Host "`n" + "=" * 70
Write-Host "All tests complete!" -ForegroundColor Green
Write-Host "=" * 70
