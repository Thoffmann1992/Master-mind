"""
BeatBazar AI Mastering Engineer v2
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from scipy.ndimage import uniform_filter
import shutil, uuid
from pathlib import Path

app = FastAPI(title="BeatBazar AI Mastering Engineer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# ── DSP ──────────────────────────────────────────────────────────────

def to_stereo(y):
    if y.ndim == 1: return np.stack([y, y], axis=1)
    if y.ndim == 2 and y.shape[0] == 2: return y.T
    return y

def to_mono(y):
    if y.ndim == 1: return y
    return np.mean(y, axis=1)

def apply_highpass(y, sr, cutoff=30, order=2):
    b, a = butter(order, np.clip(cutoff/(sr/2), 1e-4, 0.99), btype='high')
    return lfilter(b, a, y, axis=0)

def apply_shelf(y, sr, freq, gain_db, shelf_type='low'):
    if abs(gain_db) < 0.05: return y
    b, a = butter(2, np.clip(freq/(sr/2), 1e-4, 0.99), btype='low' if shelf_type=='low' else 'high')
    return y + lfilter(b, a, y, axis=0) * (10**(gain_db/20) - 1)

def apply_peaking(y, sr, freq, gain_db, q=1.4):
    if abs(gain_db) < 0.05: return y
    w0 = 2*np.pi*freq/sr; A = 10**(gain_db/40); alpha = np.sin(w0)/(2*q)
    b = np.array([(1+alpha*A)/( 1+alpha/A), (-2*np.cos(w0))/(1+alpha/A), (1-alpha*A)/(1+alpha/A)])
    a = np.array([1.0, -2*np.cos(w0)/(1+alpha/A), (1-alpha/A)/(1+alpha/A)])
    return lfilter(b, a, y, axis=0)

def compress(y, threshold_db=-18, ratio=4.0, attack_ms=10, release_ms=150, sr=44100, makeup_db=0):
    threshold = 10**(threshold_db/20); makeup = 10**(makeup_db/20)
    atk = 1-np.exp(-1/(sr*attack_ms/1000)); rel = 1-np.exp(-1/(sr*release_ms/1000))
    mono = to_mono(y); gain = np.ones(len(mono)); env = 0.0
    for i in range(len(mono)):
        lvl = abs(mono[i]); env = env+(atk if lvl>env else rel)*(lvl-env)
        gain[i] = (threshold*(env/threshold)**(1/ratio)/env) if env>threshold else 1.0
    return (y*gain*makeup) if y.ndim==1 else (y*gain[:,np.newaxis]*makeup)

def limit(y, ceiling_db=-0.3):
    ceiling = 10**(ceiling_db/20); peak = np.max(np.abs(y))
    return y*ceiling/peak if peak>ceiling else y

def stereo_width(y, width=1.2):
    y = to_stereo(y); mid=(y[:,0]+y[:,1])/2; side=(y[:,0]-y[:,1])/2*width
    y[:,0]=mid+side; y[:,1]=mid-side; return y

def normalize_lufs(y, target=-14.0):
    mono = to_mono(y); rms = np.sqrt(np.mean(mono**2)+1e-10)
    current = 20*np.log10(rms)-0.691
    return y*(10**(np.clip(target-current,-20,20)/20))

def tape_saturate(y, amount=0.3):
    drive=1+amount*4
    return (np.tanh(y*drive)/np.tanh(drive))*(1-amount*0.1)+y*amount*0.1

def noise_gate(y, sr, amount=0.3):
    if amount<=0: return y
    mono=to_mono(y); S=np.abs(librosa.stft(mono))
    thresh=np.percentile(S,15)*(1+amount*2)
    mask=uniform_filter((S>thresh).astype(float),size=(1,5))>0.5
    phase=np.angle(librosa.stft(mono))
    clean=librosa.istft(S*mask*np.exp(1j*phase))
    clean=np.pad(clean,(0,max(0,len(mono)-len(clean))))[:len(mono)]
    return np.stack([clean,clean],axis=1) if y.ndim==2 else clean

def band_db(mono, sr, low, high):
    nyq=sr/2; lo=np.clip(low/nyq,1e-4,0.99); hi=np.clip(high/nyq,1e-4,0.99)
    if lo>=hi: return -80.0
    b,a=butter(4,[lo,hi],btype='band')
    return float(20*np.log10(np.sqrt(np.mean(lfilter(b,a,mono)**2))+1e-10))

# ── 1. ANALİZ ────────────────────────────────────────────────────────

def analyze_audio(y, sr):
    mono=to_mono(y)
    tempo,_=librosa.beat.beat_track(y=mono,sr=sr)
    bpm=float(np.round(tempo,1))
    chroma=librosa.feature.chroma_cqt(y=mono,sr=sr)
    cm=np.mean(chroma,axis=1); key_idx=int(np.argmax(cm))
    notes=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    maj=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minn=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    rot=np.roll(cm,-key_idx)
    is_major=np.corrcoef(rot,maj)[0,1]>np.corrcoef(rot,minn)[0,1]
    key_full=f"{notes[key_idx]} {'Major' if is_major else 'Minor'}"
    rms=np.sqrt(np.mean(mono**2)+1e-10)
    lufs=round(20*np.log10(rms)-0.691,1)
    peak_db=round(20*np.log10(np.max(np.abs(mono))+1e-10),1)
    frms=librosa.feature.rms(y=mono,frame_length=2048,hop_length=512)[0]
    dr=round(float(20*np.log10((np.percentile(frms,95)+1e-10)/(np.percentile(frms,10)+1e-10))),1)
    if y.ndim==2 and y.shape[1]==2:
        m=(y[:,0]+y[:,1])/2; s=(y[:,0]-y[:,1])/2
        width_val=round(float(np.sqrt(np.mean(s**2)+1e-10)/( np.sqrt(np.mean(m**2)+1e-10))*100),1)
    else: width_val=0.0
    freq={'sub':band_db(mono,sr,20,60),'low':band_db(mono,sr,60,200),'mid':band_db(mono,sr,200,2000),'high':band_db(mono,sr,2000,20000)}
    genre=detect_genre(bpm,freq,dr)
    return {"bpm":bpm,"key":key_full,"lufs_current":lufs,"peak_db":peak_db,"dynamic_range":dr,"stereo_width":width_val,"freq_balance":freq,"genre":genre}

def detect_genre(bpm,freq,dr):
    sub,low,mid=freq["sub"],freq["low"],freq["mid"]
    if 125<=bpm<=145 and sub>low-5: return "Techno / EDM"
    if 60<=bpm<=100 and low>mid-3: return "Hip-Hop / Trap"
    if 100<=bpm<=130 and dr<8: return "Pop / RnB"
    if dr>12 and sub<mid-10: return "Ambient"
    return "Pop / RnB"

# ── 2. STRUKTUR ──────────────────────────────────────────────────────

def analyze_structure(y, sr):
    mono=to_mono(y); hop=512
    frms=librosa.feature.rms(y=mono,frame_length=2048,hop_length=hop)[0]
    times=librosa.frames_to_time(np.arange(len(frms)),sr=sr,hop_length=hop)
    norm=frms/(np.max(frms)+1e-10); n=len(norm); q=n//4
    labels=["Intro","Verse / Build","Chorus / Drop","Outro"]
    sections=[{"section":l,"start":round(float(times[i*q]),1),"end":round(float(times[min((i+1)*q-1,n-1)]),1),"energy":round(float(np.mean(norm[i*q:(i+1)*q]))*100,1)} for i,l in enumerate(labels)]
    return {"duration":round(len(mono)/sr,1),"sections":sections,"peak_at":round(float(times[int(np.argmax(frms))]),1),"lowest_at":round(float(times[int(np.argmin(frms))]),1)}

# ── 3. PROBLEMLƏR ────────────────────────────────────────────────────

def detect_problems(y, sr, analysis):
    mono=to_mono(y); problems=[]
    mud=band_db(mono,sr,200,400); mid=band_db(mono,sr,400,2000)
    harsh=band_db(mono,sr,3000,6000); sib=band_db(mono,sr,6000,10000)
    high=band_db(mono,sr,10000,20000); bass=band_db(mono,sr,60,200)
    if mud>mid-2: problems.append({"problem":"Mud","range":"200–400 Hz","severity":"Yüksək" if mud>=mid else "Orta","fix":f"EQ: {round(mid-mud-3,1)} dB @ 250 Hz, Q=1.5"})
    if harsh>mid+3: problems.append({"problem":"Harshness","range":"3k–6k Hz","severity":"Yüksək","fix":f"EQ: -{round(harsh-mid-2,1)} dB @ 4 kHz, Q=1.2"})
    if sib>high+4: problems.append({"problem":"Sibilance","range":"6k–10k Hz","severity":"Orta","fix":f"De-esser: -{round(sib-high-3,1)} dB @ 8 kHz"})
    clips=int(np.sum(np.abs(mono)>0.99))
    if clips>10: problems.append({"problem":"Clipping","range":"Full range","severity":"Kritik" if clips>100 else "Orta","fix":f"Gain azalt -{round(min(6,clips/100),1)} dB"})
    if y.ndim==2 and y.shape[1]==2:
        corr=float(np.corrcoef(y[:,0],y[:,1])[0,1])
        if corr<0.3: problems.append({"problem":"Stereo Phase Problem","range":"Stereo field","severity":"Yüksək","fix":"Mid/Side: Side siqnalı 20-30% azalt"})
        elif corr<0: problems.append({"problem":"Phase İnversion","range":"Stereo field","severity":"Kritik","fix":"Bir kanalı invert et"})
    if bass<mid-8: problems.append({"problem":"Bass Zəifdir","range":"60–200 Hz","severity":"Orta","fix":f"EQ: +{round(mid-bass-5,1)} dB @ 100 Hz + saturation"})
    elif bass>mid+6: problems.append({"problem":"Bass Həddindən Artıqdır","range":"60–200 Hz","severity":"Orta","fix":f"EQ: -{round(bass-mid-4,1)} dB @ 120 Hz"})
    return problems

# ── 4. GENRE TARGETS ─────────────────────────────────────────────────

GENRE_TARGETS={
    "Hip-Hop / Trap":{"lufs":-9,"lufs_range":"-8 / -10","true_peak":-0.5,"dr":6,"stereo":1.25},
    "Techno / EDM":  {"lufs":-8,"lufs_range":"-6 / -9","true_peak":-0.3,"dr":5,"stereo":1.35},
    "Pop / RnB":     {"lufs":-10,"lufs_range":"-9 / -12","true_peak":-1.0,"dr":8,"stereo":1.15},
    "Ambient":       {"lufs":-14,"lufs_range":"-12 / -16","true_peak":-1.0,"dr":14,"stereo":1.4},
}

def get_target(genre): return GENRE_TARGETS.get(genre,GENRE_TARGETS["Pop / RnB"])

# ── 5. CHAIN BUILDER ─────────────────────────────────────────────────

def build_chain(genre, problems, analysis):
    target=get_target(genre)
    eq_fixes=[]
    for p in problems:
        if p["problem"]=="Mud": eq_fixes.append({"freq":250,"gain":-2.5,"q":1.5,"note":"Mud cut"})
        if p["problem"]=="Harshness": eq_fixes.append({"freq":4000,"gain":-2.0,"q":1.2,"note":"Harshness cut"})
        if p["problem"]=="Sibilance": eq_fixes.append({"freq":8000,"gain":-2.5,"q":1.0,"note":"De-esser"})
        if p["problem"]=="Bass Zəifdir": eq_fixes.append({"freq":100,"gain":2.5,"q":1.0,"note":"Bass boost"})
        if p["problem"]=="Bass Həddindən Artıqdır": eq_fixes.append({"freq":120,"gain":-2.5,"q":1.0,"note":"Bass cut"})
    comp_map={"Hip-Hop / Trap":{"threshold":-14,"ratio":6,"attack":5,"release":80,"makeup":3},"Techno / EDM":{"threshold":-12,"ratio":7,"attack":3,"release":60,"makeup":4},"Pop / RnB":{"threshold":-18,"ratio":3.5,"attack":12,"release":180,"makeup":2},"Ambient":{"threshold":-22,"ratio":2.0,"attack":25,"release":400,"makeup":2}}
    return {"corrective_eq":eq_fixes,"multiband_comp":comp_map.get(genre,comp_map["Pop / RnB"]),"saturation":0.2 if genre in ["Hip-Hop / Trap","Techno / EDM"] else 0.15,"stereo_width":target["stereo"],"dynamic_eq":{"low_cut":30,"air_boost":1.5},"limiter":{"ceiling":target["true_peak"],"target_lufs":target["lufs"]}}

# ── 6. MASTERING TƏTBİQİ ─────────────────────────────────────────────

def apply_mastering(y, sr, chain, noise_reduction=0.0):
    if noise_reduction>0: y=noise_gate(y,sr,noise_reduction)
    y=apply_highpass(y,sr,cutoff=chain["dynamic_eq"]["low_cut"])
    for eq in chain["corrective_eq"]: y=apply_peaking(y,sr,eq["freq"],eq["gain"],eq.get("q",1.4))
    y=apply_shelf(y,sr,10000,chain["dynamic_eq"]["air_boost"],'high')
    c=chain["multiband_comp"]
    y=compress(y,threshold_db=c["threshold"],ratio=c["ratio"],attack_ms=c["attack"],release_ms=c["release"],sr=sr,makeup_db=c["makeup"])
    y=tape_saturate(y,amount=chain["saturation"])
    y=stereo_width(y,width=chain["stereo_width"])
    y=normalize_lufs(y,target=chain["limiter"]["target_lufs"])
    y=limit(y,ceiling_db=chain["limiter"]["ceiling"])
    return y

# ── 7. BANDLAB STEPS ─────────────────────────────────────────────────

def bandlab_steps(chain, problems, genre):
    n=1; steps=[]
    steps.append({"step":n,"name":"High-Pass Filter","how":f"FX → EQ → High-Pass @ {chain['dynamic_eq']['low_cut']} Hz","why":"Rumble təmizlə"}); n+=1
    for eq in chain["corrective_eq"]:
        steps.append({"step":n,"name":f"EQ — {eq['note']}","how":f"FX → EQ → Peaking: {eq['gain']:+.1f} dB @ {eq['freq']} Hz, Q={eq['q']}","why":eq["note"]}); n+=1
    steps.append({"step":n,"name":"Air Boost","how":f"FX → EQ → High Shelf: +{chain['dynamic_eq']['air_boost']} dB @ 10 kHz","why":"Parlaqlıq əlavə et"}); n+=1
    c=chain["multiband_comp"]
    steps.append({"step":n,"name":"Compressor","how":f"FX → Compressor → Thr: {c['threshold']} dB | Ratio: {c['ratio']}:1 | Atk: {c['attack']}ms | Rel: {c['release']}ms | Makeup: +{c['makeup']} dB","why":"Dinamika idarəsi"}); n+=1
    steps.append({"step":n,"name":"Stereo Width","how":f"FX → Stereo Tool → Width: {int(chain['stereo_width']*100)}%","why":"Stereo genişləndir"}); n+=1
    steps.append({"step":n,"name":"Limiter","how":f"Master FX → Limiter → Ceiling: {chain['limiter']['ceiling']} dBTP | {chain['limiter']['target_lufs']} LUFS","why":"Klipingi önlə"}); n+=1
    return steps

# ── 8. FL STUDIO MOBILE STEPS ────────────────────────────────────────

def fl_mobile_steps(chain, genre):
    n=1; steps=[]
    eq_bands=" | ".join([f"{e['gain']:+.1f}dB@{e['freq']}Hz" for e in chain["corrective_eq"]]) or "Korrektiv band yoxdur"
    steps.append({"step":n,"name":"Parametric EQ 2","plugin":"Mixer → Master → Parametric EQ 2","settings":f"HP @ {chain['dynamic_eq']['low_cut']}Hz | {eq_bands} | HighShelf +{chain['dynamic_eq']['air_boost']}dB@10kHz"}); n+=1
    c=chain["multiband_comp"]
    steps.append({"step":n,"name":"Fruity Compressor","plugin":"Mixer → Master → Fruity Compressor","settings":f"Thr:{c['threshold']}dB | {c['ratio']}:1 | Atk:{c['attack']}ms | Rel:{c['release']}ms | +{c['makeup']}dB"}); n+=1
    steps.append({"step":n,"name":"Fruity Fast Dist (Saturation)","plugin":"Mixer → Master → Fast Dist","settings":f"Soft Clip | Amount:{int(chain['saturation']*100)}% | Mix:30%"}); n+=1
    steps.append({"step":n,"name":"Stereo Shaper","plugin":"Mixer → Master → Stereo Shaper","settings":f"Width:{int(chain['stereo_width']*100)}% | Mono below 120Hz"}); n+=1
    steps.append({"step":n,"name":"Maximus","plugin":"Mixer → Master → Maximus","settings":f"Ceiling:{chain['limiter']['ceiling']}dBTP | {chain['limiter']['target_lufs']}LUFS | Rel:50ms"}); n+=1
    return steps

# ── 9. QİYMƏT ────────────────────────────────────────────────────────

def score_track(analysis, problems, target):
    lufs_diff=abs(analysis["lufs_current"]-target["lufs"])
    loudness=max(0,100-lufs_diff*8)
    freq=analysis["freq_balance"]
    balance_diff=abs(freq["low"]-freq["mid"])+abs(freq["mid"]-freq["high"])
    clarity=max(0,100-balance_diff*2)-sum(10 for p in problems if p["problem"] in ["Harshness","Sibilance","Mud"])
    stereo=min(100,analysis["stereo_width"]*1.5)
    if any(p["problem"] in ["Stereo Phase Problem","Phase İnversion"] for p in problems): stereo=max(0,stereo-30)
    overall=(loudness*0.35+clarity*0.35+stereo*0.3)
    return {"loudness":round(min(100,max(0,loudness)),1),"clarity":round(min(100,max(0,clarity)),1),"stereo":round(min(100,max(0,stereo)),1),"overall":round(min(100,max(0,overall)),1)}

# ── 10. EXPORT ───────────────────────────────────────────────────────

EXPORT_OPTIONS=[
    {"format":"WAV 24-bit / 44.1 kHz","use":"Streaming, mastering arxiv","best_for":"Əksər hallarda ən yaxşı seçim","recommended":True},
    {"format":"WAV 24-bit / 48 kHz","use":"Video, film, YouTube","best_for":"Video kontentlə istifadə üçün","recommended":False},
    {"format":"MP3 320 kbps","use":"SoundCloud, sosial media","best_for":"Fayl ölçüsü vacibdirsə","recommended":False},
    {"format":"AAC 256 kbps","use":"Apple Music, iPhone","best_for":"Apple platformaları üçün","recommended":False},
]

# ── 11. REPORT FORMATTER ─────────────────────────────────────────────

def format_report(analysis, structure, problems, target, chain, scores, bandlab, fl_steps):
    freq=analysis["freq_balance"]
    prob_lines="\n".join([f"  • {p['problem']} ({p['range']}) [{p['severity']}]\n    → {p['fix']}" for p in problems]) or "  ✅ Ciddi problem aşkarlanmadı"
    struct_lines="\n".join([f"  {s['section']}: {s['start']}s–{s['end']}s | Enerji:{s['energy']}%" for s in structure["sections"]])
    bl_lines="\n".join([f"  {s['step']}. {s['name']}\n     ↳ {s['how']}\n     💡 {s['why']}" for s in bandlab])
    fl_lines="\n".join([f"  {s['step']}. {s['name']}\n     Plugin: {s['plugin']}\n     ⚙️  {s['settings']}" for s in fl_steps])
    exp_lines="\n".join([f"  {'✅' if e['recommended'] else '•'} {e['format']}\n    {e['use']} — {e['best_for']}" for e in EXPORT_OPTIONS])
    return f"""
╔══════════════════════════════════════════════════════╗
║       🎛️  BeatBazar AI Mastering Engineer           ║
╚══════════════════════════════════════════════════════╝

📊 ANALİZ
  BPM: {analysis['bpm']}  |  Key: {analysis['key']}  |  Janr: {analysis['genre']}
  LUFS (mövcud): {analysis['lufs_current']}  |  Target: {target['lufs_range']}
  Peak: {analysis['peak_db']} dBFS  |  DR: {analysis['dynamic_range']}  |  Stereo: {analysis['stereo_width']}%
  Freq → Sub:{freq['sub']}dB | Low:{freq['low']}dB | Mid:{freq['mid']}dB | High:{freq['high']}dB

🏗️ STRUKTUR  ({structure['duration']}s)
{struct_lines}
  🔺 Pik: {structure['peak_at']}s  |  🔻 Ən sakit: {structure['lowest_at']}s

⚠️ PROBLEMLƏR
{prob_lines}

🛠 HƏLLLƏR (Mastering Chain)
  1. Corrective EQ  → {len(chain['corrective_eq'])} band
  2. Multiband Comp → {chain['multiband_comp']['threshold']}dB thr | {chain['multiband_comp']['ratio']}:1
  3. Saturation     → {int(chain['saturation']*100)}% tape drive
  4. Stereo Imaging → {int(chain['stereo_width']*100)}%
  5. Dynamic EQ     → Air +{chain['dynamic_eq']['air_boost']}dB @ 10kHz
  6. Limiter        → {chain['limiter']['ceiling']}dBTP | {chain['limiter']['target_lufs']}LUFS

🎛️ BANDLAB STEPS
{bl_lines}

🎚️ FL STUDIO MOBILE STEPS
{fl_lines}

📈 QİYMƏTLƏNDİRMƏ
  Loudness: {scores['loudness']}/100  |  Clarity: {scores['clarity']}/100
  Stereo: {scores['stereo']}/100  |  Overall: {scores['overall']}/100

📥 EXPORT
{exp_lines}
""".strip()

# ── ENDPOINTS ────────────────────────────────────────────────────────

@app.get("/health")
def health(): return {"status":"ok","engine":"BeatBazar AI Mastering Engineer v2"}

@app.post("/analyze")
async def analyze_only(file: UploadFile=File(...)):
    uid=str(uuid.uuid4())[:8]; suffix=Path(file.filename or "a.mp3").suffix or ".mp3"
    inp=UPLOAD_DIR/f"{uid}{suffix}"
    try:
        with inp.open("wb") as buf: shutil.copyfileobj(file.file,buf)
        y,sr=librosa.load(str(inp),sr=None,mono=False); y=to_stereo(y)
        analysis=analyze_audio(y,sr); structure=analyze_structure(y,sr)
        problems=detect_problems(y,sr,analysis); target=get_target(analysis["genre"])
        scores=score_track(analysis,problems,target)
        return JSONResponse({"analysis":analysis,"structure":structure,"problems":problems,"target":target,"scores":scores,"export_options":EXPORT_OPTIONS})
    finally:
        if inp.exists(): inp.unlink()

@app.post("/report")
async def get_report(file: UploadFile=File(...), profile: str=Form("auto")):
    uid=str(uuid.uuid4())[:8]; suffix=Path(file.filename or "a.mp3").suffix or ".mp3"
    inp=UPLOAD_DIR/f"{uid}{suffix}"
    try:
        with inp.open("wb") as buf: shutil.copyfileobj(file.file,buf)
        y,sr=librosa.load(str(inp),sr=None,mono=False); y=to_stereo(y)
        analysis=analyze_audio(y,sr); structure=analyze_structure(y,sr)
        problems=detect_problems(y,sr,analysis); genre=analysis["genre"]
        target=get_target(genre); chain=build_chain(genre,problems,analysis)
        scores=score_track(analysis,problems,target)
        bl=bandlab_steps(chain,problems,genre); fl=fl_mobile_steps(chain,genre)
        report=format_report(analysis,structure,problems,target,chain,scores,bl,fl)
        return JSONResponse({"report":report,"analysis":analysis,"structure":structure,"problems":problems,"scores":scores,"bandlab_steps":bl,"fl_mobile_steps":fl,"export_options":EXPORT_OPTIONS})
    finally:
        if inp.exists(): inp.unlink()

@app.post("/master")
async def master_audio(file: UploadFile=File(...), profile: str=Form("auto"), noise_reduction: float=Form(0.0), sample_rate: int=Form(44100)):
    uid=str(uuid.uuid4())[:8]; suffix=Path(file.filename or "a.mp3").suffix or ".mp3"
    inp=UPLOAD_DIR/f"{uid}{suffix}"; out=PROCESSED_DIR/f"{uid}_master.wav"
    try:
        with inp.open("wb") as buf: shutil.copyfileobj(file.file,buf)
        y,sr=librosa.load(str(inp),sr=None,mono=False); y=to_stereo(y)
        if sample_rate!=sr:
            mono_rs=librosa.resample(to_mono(y),orig_sr=sr,target_sr=sample_rate)
            y=np.stack([mono_rs,mono_rs],axis=1); sr=sample_rate
        analysis=analyze_audio(y,sr); structure=analyze_structure(y,sr)
        problems=detect_problems(y,sr,analysis)
        pm={"streaming":"Pop / RnB","club":"Techno / EDM","warm":"Hip-Hop / Trap","cinematic":"Ambient"}
        genre=pm.get(profile,analysis["genre"]) if profile!="auto" else analysis["genre"]
        target=get_target(genre); chain=build_chain(genre,problems,analysis)
        scores=score_track(analysis,problems,target)
        bl=bandlab_steps(chain,problems,genre); fl=fl_mobile_steps(chain,genre)
        report=format_report(analysis,structure,problems,target,chain,scores,bl,fl)
        y_m=np.clip(apply_mastering(y.copy(),sr,chain,noise_reduction),-1.0,1.0)
        sf.write(str(out),y_m,sr,subtype="PCM_24")
        stem=Path(file.filename or "audio").stem
        return FileResponse(path=str(out),filename=f"{stem}_beatbazar_master.wav",media_type="audio/wav",headers={"X-Score":str(scores["overall"]),"X-Genre":genre,"X-LUFS":str(target["lufs"]),"X-Report":report[:300]})
    except Exception as e: raise HTTPException(500,f"Xəta: {e}")
    finally:
        if inp.exists(): inp.unlink()

@app.get("/",response_class=HTMLResponse)
async def root():
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>BeatBazar AI Mastering</title>
<style>body{font-family:monospace;background:#080810;color:#e2ff4e;padding:40px;line-height:1.8}h1{margin-bottom:4px}.ep{background:#0d0d1a;border:1px solid #1c1c30;border-radius:6px;padding:10px 14px;margin:6px 0}.method{color:#ff4d6d;margin-right:8px}.desc{color:#555;font-size:11px;margin-top:2px}</style>
</head><body>
<h1>🎛️ BeatBazar AI Mastering Engineer</h1>
<p style="color:#666">Professional mastering API — 11 funksiya</p>
<div class="ep"><span class="method">GET</span>/health<div class="desc">Status</div></div>
<div class="ep"><span class="method">POST</span>/analyze<div class="desc">BPM · Key · LUFS · Problemlər · Qiymət</div></div>
<div class="ep"><span class="method">POST</span>/report<div class="desc">Tam text report (BandLab + FL Mobile steps)</div></div>
<div class="ep"><span class="method">POST</span>/master<div class="desc">Tam mastering + audio endir (profile: auto|streaming|club|warm|cinematic)</div></div>
</body></html>""")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000,reload=False)
