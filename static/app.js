// ===== MedAI Clinical Portal - Client-Side JavaScript =====

// ===== AUTH PAGE LOGIC =====
function switchAuthTab(tab) {
    document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.auth-form').forEach(f => f.classList.add('hidden'));
    
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`form-${tab}`).classList.remove('hidden');
}

async function handleLogin(e) {
    e.preventDefault();
    const msgEl = document.getElementById('login-msg');
    msgEl.innerHTML = '';
    
    const email_phone = document.getElementById('login-email').value.trim();
    const password = document.getElementById('login-password').value;
    
    if (!email_phone || !password) {
        msgEl.innerHTML = '<div class="alert alert-error">⚠️ Please fill in all fields.</div>';
        return;
    }
    
    const btn = e.target.querySelector('button[type="submit"]');
    btn.innerHTML = '<span class="spinner"></span> Signing In...';
    btn.disabled = true;
    
    try {
        const res = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email_phone, password })
        });
        const data = await res.json();
        
        if (data.success) {
            msgEl.innerHTML = `<div class="alert alert-success">👋 Welcome back, ${data.user.name}!</div>`;
            setTimeout(() => window.location.href = '/dashboard', 600);
        } else {
            msgEl.innerHTML = `<div class="alert alert-error">❌ ${data.message}</div>`;
        }
    } catch (err) {
        msgEl.innerHTML = '<div class="alert alert-error">⚠️ Server error. Please try again.</div>';
    }
    
    btn.innerHTML = 'Sign In to Patient File';
    btn.disabled = false;
}

async function handleRegister(e) {
    e.preventDefault();
    const msgEl = document.getElementById('register-msg');
    msgEl.innerHTML = '';
    
    const name = document.getElementById('reg-name').value.trim();
    const email_phone = document.getElementById('reg-email').value.trim();
    const password = document.getElementById('reg-password').value;
    const age = parseInt(document.getElementById('reg-age').value);
    const gender = document.getElementById('reg-gender').value;
    const pre_existing = document.getElementById('reg-conditions').value.trim();
    
    if (!name || !email_phone || !password) {
        msgEl.innerHTML = '<div class="alert alert-error">⚠️ Name, Email/Phone, and Password are required.</div>';
        return;
    }
    if (password.length < 6) {
        msgEl.innerHTML = '<div class="alert alert-error">⚠️ Password must be at least 6 characters.</div>';
        return;
    }
    
    const btn = e.target.querySelector('button[type="submit"]');
    btn.innerHTML = '<span class="spinner"></span> Creating File...';
    btn.disabled = true;
    
    try {
        const res = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email_phone, password, age, gender, pre_existing })
        });
        const data = await res.json();
        
        if (data.success) {
            msgEl.innerHTML = '<div class="alert alert-success">✅ Patient file created! Please sign in.</div>';
            setTimeout(() => switchAuthTab('signin'), 1500);
        } else {
            msgEl.innerHTML = `<div class="alert alert-error">❌ ${data.message}</div>`;
        }
    } catch (err) {
        msgEl.innerHTML = '<div class="alert alert-error">⚠️ Server error. Please try again.</div>';
    }
    
    btn.innerHTML = 'Create Diagnostic File';
    btn.disabled = false;
}

// ===== DASHBOARD MODULE SWITCHING =====
function switchModule(mod) {
    document.querySelectorAll('.module-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.module-panel').forEach(p => p.classList.add('hidden'));
    
    document.querySelector(`[data-module="${mod}"]`).classList.add('active');
    document.getElementById(`panel-${mod}`).classList.remove('hidden');
}

// ===== HEART DISEASE PREDICTION =====
async function analyzeHeart(e) {
    e.preventDefault();
    const resultEl = document.getElementById('heart-result');
    resultEl.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Analyzing cardiovascular biomarkers...</div>';
    
    const formData = {
        age: parseInt(document.getElementById('h-age').value),
        sex: document.getElementById('h-sex').value,
        resting_bp: parseInt(document.getElementById('h-bp').value),
        cholesterol: parseInt(document.getElementById('h-chol').value),
        fasting_bs: parseInt(document.getElementById('h-fbs').value),
        max_hr: parseInt(document.getElementById('h-hr').value),
        exercise_angina: document.getElementById('h-angina').value,
        chest_pain: document.getElementById('h-chest').value,
        resting_ecg: document.getElementById('h-ecg').value,
        oldpeak: parseFloat(document.getElementById('h-oldpeak').value),
        st_slope: document.getElementById('h-slope').value,
    };
    
    try {
        const res = await fetch('/api/predict/heart', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const data = await res.json();
        renderHeartResult(data, resultEl);
    } catch (err) {
        resultEl.innerHTML = '<div class="alert alert-error">⚠️ Prediction failed. Check server.</div>';
    }
}

function renderHeartResult(data, el) {
    const isHigh = data.prediction === 1;
    const pct = (data.probability * 100).toFixed(1);
    const barClass = isHigh ? 'red' : 'green';
    
    const precautions = isHigh ? `
        <div class="guide-card" style="border-color: #ef4444;">
            <h4 style="color: #f87171;">🚨 Critical Precautions</h4>
            <ul>
                <li><strong>Medical Consultation:</strong> Schedule an immediate consultation with a cardiologist.</li>
                <li><strong>Diagnostic Screening:</strong> Suggest a cardiac stress test, echocardiogram, or CT coronary angiogram.</li>
                <li><strong>Dietary Shift:</strong> Adopt a strict low-sodium, low-cholesterol Mediterranean diet (&lt;1,500mg sodium/day).</li>
                <li><strong>Activity:</strong> Restrict sudden heavy lifting. Opt for light walking.</li>
                <li><strong>Vitals:</strong> Monitor blood pressure and resting heart rate daily.</li>
            </ul>
        </div>
    ` : `
        <div class="guide-card" style="border-color: #10b981;">
            <h4 style="color: #34d399;">🛡️ Preventative Measures</h4>
            <ul>
                <li><strong>Exercise:</strong> Maintain 150 mins of moderate aerobic activity per week.</li>
                <li><strong>Nutrition:</strong> Keep a high-fiber diet rich in whole grains and omega-3.</li>
                <li><strong>Vitals:</strong> Annual serum cholesterol & lipid profile monitoring.</li>
                <li><strong>Stress:</strong> Daily mindfulness or relaxation breathing exercises.</li>
            </ul>
        </div>
    `;
    
    el.innerHTML = `
        <div class="grid-2">
            <div>
                <h3 class="mb-2" style="color: #e5e7eb;">Risk Analysis Report</h3>
                <div class="result-card ${isHigh ? 'result-high' : 'result-low'}">
                    <h3>${isHigh ? '⚠️ Elevated Risk Detected' : '🟢 Normal Risk Detected'}</h3>
                    <p>${isHigh ? 'Patient shows cardiovascular biomarkers indicating high susceptibility.' : 'Patient indicators are within healthy clinical boundaries.'}</p>
                    <p class="mt-2"><strong>Risk Probability:</strong> ${pct}%</p>
                    <div class="progress-bar"><div class="progress-fill ${barClass}" style="width: ${pct}%"></div></div>
                </div>
            </div>
            <div>
                <h3 class="mb-2" style="color: #e5e7eb;">Clinical Precautions</h3>
                ${precautions}
            </div>
        </div>
    `;
}

// ===== DIABETES PREDICTION =====
function togglePregnancy() {
    const sex = document.getElementById('d-sex').value;
    const pregGroup = document.getElementById('preg-group');
    if (sex === 'Male') {
        pregGroup.classList.add('hidden');
        document.getElementById('d-preg').value = 0;
    } else {
        pregGroup.classList.remove('hidden');
    }
}

async function analyzeDiabetes(e) {
    e.preventDefault();
    const resultEl = document.getElementById('diabetes-result');
    resultEl.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Evaluating diabetic risk levels...</div>';
    
    const formData = {
        pregnancies: parseInt(document.getElementById('d-preg').value),
        glucose: parseInt(document.getElementById('d-glucose').value),
        bp: parseInt(document.getElementById('d-bp').value),
        skin: parseInt(document.getElementById('d-skin').value),
        insulin: parseInt(document.getElementById('d-insulin').value),
        bmi: parseFloat(document.getElementById('d-bmi').value),
        dpf: parseFloat(document.getElementById('d-dpf').value),
        age: parseInt(document.getElementById('d-age').value),
    };
    
    try {
        const res = await fetch('/api/predict/diabetes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        const data = await res.json();
        renderDiabetesResult(data, formData.glucose, resultEl);
    } catch (err) {
        resultEl.innerHTML = '<div class="alert alert-error">⚠️ Prediction failed. Check server.</div>';
    }
}

function renderDiabetesResult(data, glucose, el) {
    const isHigh = data.prediction === 1;
    const pct = (data.probability * 100).toFixed(1);
    const barClass = isHigh ? 'red' : 'green';
    
    let dietHTML = '';
    if (glucose >= 160 || (isHigh && data.probability >= 0.70)) {
        dietHTML = `
            <div class="guide-card" style="border-color: #ef4444;">
                <h4 style="color: #f87171;">🔴 Diet: Severe Hyperglycemia</h4>
                <div class="diet-grid">
                    <div class="diet-eat"><h5>🥗 EAT THIS</h5><ul><li>Leafy greens (spinach, kale)</li><li>Non-starchy veggies (broccoli, karela)</li><li>Healthy fats (avocado, walnuts)</li><li>Lean proteins (egg whites, tofu)</li></ul></div>
                    <div class="diet-avoid"><h5>🚫 AVOID THIS</h5><ul><li>Sweets, sugar, honey, jaggery</li><li>Refined carbs (white rice, maida)</li><li>Sugary beverages (soda, juices)</li><li>High-sugar fruits (mangoes)</li></ul></div>
                </div>
            </div>`;
    } else if ((glucose >= 120 && glucose < 160) || (isHigh && data.probability < 0.70)) {
        dietHTML = `
            <div class="guide-card" style="border-color: #fb923c;">
                <h4 style="color: #fb923c;">🟡 Diet: Moderate Sugar Control</h4>
                <div class="diet-grid">
                    <div class="diet-eat"><h5>🥗 EAT THIS</h5><ul><li>Whole grains (quinoa, oats, ragi)</li><li>Legumes, beans, sprouts</li><li>Greek yogurt (unsweetened)</li><li>Low-sugar berries</li></ul></div>
                    <div class="diet-avoid"><h5>🚫 AVOID THIS</h5><ul><li>Refined cereals</li><li>Dried fruits (raisins, dates)</li><li>Potato, sweet potato</li><li>Sweetened milk products</li></ul></div>
                </div>
            </div>`;
    } else {
        dietHTML = `
            <div class="guide-card" style="border-color: #10b981;">
                <h4 style="color: #34d399;">🟢 Diet: Healthy Maintenance</h4>
                <div class="diet-grid">
                    <div class="diet-eat"><h5>🥗 EAT THIS</h5><ul><li>Balanced proteins & complex carbs</li><li>High-fiber fruits (apples, guava)</li><li>Green tea, unsweetened drinks</li></ul></div>
                    <div class="diet-avoid"><h5>🚫 LIMIT THIS</h5><ul><li>Processed junk foods</li><li>Excessive salt & saturated fats</li><li>Soda and sugary desserts</li></ul></div>
                </div>
            </div>`;
    }
    
    el.innerHTML = `
        <div class="grid-2">
            <div>
                <h3 class="mb-2" style="color: #e5e7eb;">Risk Analysis Report</h3>
                <div class="result-card ${isHigh ? 'result-high' : 'result-low'}">
                    <h3>${isHigh ? '⚠️ High Diabetes Risk' : '🟢 Normal Diabetes Risk'}</h3>
                    <p>${isHigh ? 'High probability of diabetes detected. Recommend glucose tolerance profiling.' : 'Normal biological values with low diabetes risk.'}</p>
                    <p class="mt-2"><strong>Diabetic Probability:</strong> ${pct}%</p>
                    <div class="progress-bar"><div class="progress-fill ${barClass}" style="width: ${pct}%"></div></div>
                </div>
            </div>
            <div>
                <h3 class="mb-2" style="color: #e5e7eb;">Personalized Diet Guide</h3>
                ${dietHTML}
            </div>
        </div>
    `;
}

// ===== LOAD HISTORY =====
async function loadHistory() {
    const el = document.getElementById('history-content');
    el.innerHTML = '<div class="alert alert-info"><span class="spinner"></span> Loading diagnostic records...</div>';
    
    try {
        const res = await fetch('/api/history');
        const data = await res.json();
        
        if (!data.reports || data.reports.length === 0) {
            el.innerHTML = '<div class="alert alert-info">ℹ️ No diagnostic records found. Run a report to start building your health file.</div>';
            return;
        }
        
        let rows = '';
        data.reports.forEach(r => {
            const isHigh = r.result.includes('Elevated') || r.result.includes('High');
            const badgeClass = isHigh ? 'badge-danger' : 'badge-success';
            const icon = isHigh ? '⚠️' : '🟢';
            rows += `
                <tr>
                    <td>${r.timestamp}</td>
                    <td>${r.module}</td>
                    <td><span class="badge ${badgeClass}">${icon} ${r.result}</span></td>
                    <td>${(r.probability * 100).toFixed(1)}%</td>
                    <td style="font-size: 12px; color: #9ca3af;">${r.details}</td>
                </tr>
            `;
        });
        
        el.innerHTML = `
            <div class="card" style="overflow-x: auto;">
                <div class="card-header">Patient Health Record Timeline</div>
                <table class="history-table">
                    <thead><tr><th>Date</th><th>Module</th><th>Result</th><th>Risk</th><th>Details</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    } catch (err) {
        el.innerHTML = '<div class="alert alert-error">⚠️ Failed to load records.</div>';
    }
}
