"""Office VLM — Compact with FIXED color detection"""
import time, cv2, numpy as np, pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from shapely.geometry import Point, Polygon

try: import webcolors; _HAS_WC = True; CSS3 = webcolors.CSS3_HEX_TO_NAMES
except: _HAS_WC = False

CFG = {"model":"yolov8n.pt", "video":"test_office_video.mp4", "w":1280, "h":720, "sz":640, "conf":0.45, "iou":0.5, "csv":"movement_events.csv"}
TRACKER = DeepSort(max_age=30, n_init=3, max_iou_distance=0.8, max_cosine_distance=0.4, nn_budget=50)
ZONES = {n:Polygon(p) for n,p in [("Reception",[(0,0),(426,0),(426,720),(0,720)]),("Office",[(427,0),(853,0),(853,720),(427,720)]),("Meeting Room",[(854,0),(1280,0),(1280,720),(854,720)])]}
STATE = {"last_zone":{}, "visits":{}, "profiles":{}, "move_hist":{}, "zone_state":{}, "events":[]}

def open_cap():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened(): print(f"Camera {i} OK"); return cap
        cap.release()
    print(f"Fallback: {CFG['video']}")
    cap = cv2.VideoCapture(CFG['video'])
    if not cap.isOpened(): raise RuntimeError("No camera/video")
    return cap

def color_name(b,g,r):
    br,sat = (r+g+b)/3, (max(r,g,b)-min(r,g,b))/max(r,g,b) if max(r,g,b)>0 else 0
    if br>200 and sat<0.15: return "White"
    if 160<br<200 and sat<0.15: return "Light Gray"
    if 80<br<160 and sat<0.20: return "Gray"
    if br<50: return "Black"
    if 50<br<80 and sat<0.20: return "Dark Gray"
    if 140<br<200 and 0.10<sat<0.35 and r>g>b and (r-b)<60: return "Beige"
    if 60<br<140 and r>g>b and (r-g)<40 and (g-b)>10: return "Brown"
    if sat>0.25:
        if r>max(g,b)*1.3 and r>140: return "Bright Red" if r>200 and br>160 else "Red"
        if r>150 and g>100 and b<100 and r>g: return "Orange"
        if r>150 and g>150 and b<130: return "Yellow"
        if g>max(r,b)*1.2 and g>100: return "Green"
        if b>max(r,g)*1.2 and b>100: return "Blue"
        if b>120 and g>120 and r<100: return "Cyan"
        if r>100 and b>100 and g<min(r,b)*0.7: return "Purple"
        if r>150 and g>100 and b>100 and r>g: return "Pink"
    if 0.15<sat<0.35:
        if r>g and r>b: return "Light Pink" if br>160 else "Dusty Rose"
        if g>r and g>b: return "Mint" if br>160 else "Olive"
        if b>r and b>g: return "Sky Blue" if br>160 else "Navy"
    return "Mixed"

def get_clothing(frm, box):
    x1,y1,x2,y2 = [max(0,min(s,int(v))) for v,s in zip(box,[frm.shape[1],frm.shape[0],frm.shape[1],frm.shape[0]])]
    if x2<=x1 or y2<=y1: return "Unknown"
    crop = frm[y1:y2,x1:x2]
    if crop.size==0: return "Unknown"
    h = crop.shape[0]
    top = crop[int(h*0.2):int(h*0.6),:] if h>0 else crop
    if top.size==0: top = crop
    top = cv2.resize(top,(32,32))
    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab[:,:,0], 30, 240)
    avg = cv2.mean(top,mask=mask)[:3] if cv2.countNonZero(mask)>0 else cv2.mean(top)[:3]
    if _HAS_WC:
        try:
            rgb = (int(avg[2]),int(avg[1]),int(avg[0]))
            dist = {sum((rgb[i]-webcolors.hex_to_rgb(h)[i])**2 for i in range(3)):n for h,n in CSS3.items()}
            return dist[min(dist)].replace('grey','gray').title()
        except: pass
    return color_name(int(avg[0]),int(avg[1]),int(avg[2]))

def get_behavior(tid,x1,y1,x2,y2):
    cx,cy = (x1+x2)/2,(y1+y2)/2
    h = STATE["move_hist"].setdefault(tid,{"pos":[],"spd":[],"zig":0,"pse":0,"dir":None,"stb":0})
    h["pos"].append((cx,cy))
    if len(h["pos"])>30: h["pos"].pop(0)
    if len(h["pos"])<3: return "Standing"
    prv = h["pos"][-2]
    dst = np.hypot(cx-prv[0],cy-prv[1])
    h["spd"].append(dst)
    if len(h["spd"])>30: h["spd"].pop(0)
    if dst>2:
        dx,dy = cx-prv[0],cy-prv[1]
        cdir = np.arctan2(dy,dx)
        if h["dir"] is not None:
            adiff = abs(cdir-h["dir"])
            if adiff>np.pi: adiff = 2*np.pi-adiff
            if adiff>np.pi/2.5: h["zig"]+=1
        h["dir"],h["stb"] = cdir,0
    else: h["stb"]+=1
    asp = np.mean(h["spd"][-8:]) if len(h["spd"])>=3 else 0
    var = np.std(h["spd"][-8:]) if len(h["spd"])>=3 else 0
    h["pse"] = h["pse"]+1 if asp<2 and h["stb"]>5 else max(0,h["pse"]-1)
    dx_t,dy_t = h["pos"][-1][0]-h["pos"][0][0],h["pos"][-1][1]-h["pos"][0][1]
    td = np.hypot(dx_t,dy_t)
    if td>50 and var<8:
        if abs(dx_t)>abs(dy_t)*1.5: return "Walking Right" if dx_t>0 else "Walking Left"
        if abs(dy_t)>abs(dx_t)*1.5: return "Walking Down" if dy_t>0 else "Walking Up"
    if asp>28 and h["zig"]>4: return "Angry/Agitated"
    if asp>22: return "Rushing/Hurried"
    if asp>14 and h["zig"]>3 and var>6: return "Anxious/Nervous"
    if asp<8 and h["zig"]>5 and var>4: return "Confused/Lost"
    if 12<asp<20 and h["zig"]<2 and var<5: return "Happy/Energetic"
    if 6<asp<16 and h["pse"]>4 and h["zig"]<3: return "Curious/Exploring"
    if 4<asp<10 and h["stb"]<3: return "Calm/Relaxed"
    if asp<3.5 and h["pse"]>6: return "Sad/Tired"
    if 9<asp<16 and h["zig"]<2 and var<4: return "Focused/Determined"
    if asp<6 and h["pse"]>3 and h["zig"]<2: return "Cautious/Careful"
    if h["stb"]>8: return "Standing"
    return "Normal"

def update_prof(tid,cloth,beh):
    p = STATE["profiles"].setdefault(tid,{"clothing":cloth,"behaviors":[beh],"first_seen":datetime.now().strftime("%H:%M:%S")})
    if beh not in p["behaviors"]: p["behaviors"].append(beh)
    return p

def zone_status(tid,zn):
    s = STATE["zone_state"].setdefault(tid,{"zone":zn,"changes":0,"time":time.time(),"reset":time.time()})
    now = time.time()
    if now-s["reset"]>60: s["changes"],s["reset"]=0,now
    if s["zone"]!=zn: s["changes"]+=1; s["zone"],s["time"]=zn,now
    return "Unusual" if s["changes"]>5 else "Stationary" if now-s["time"]>120 else "Normal"

def main():
    print(f"Office VLM | Model: {CFG['model']}")
    mdl = YOLO(CFG['model'])
    cap = open_cap()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CFG['w']); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CFG['h'])
    while True:
        ret,frm = cap.read()
        if not ret: print("Stream ended"); break
        frm = cv2.resize(frm,(CFG['w'],CFG['h']))
        res = mdl(frm,conf=CFG['conf'],iou=CFG['iou'],imgsz=CFG['sz'],verbose=False)
        dets = []
        for r in res:
            bxs,cfs,cls = getattr(r.boxes,"xyxy",None),getattr(r.boxes,"conf",None),getattr(r.boxes,"cls",None)
            if bxs is None: continue
            bxs,cfs,cls = (x.cpu().numpy() if hasattr(x,"cpu") else np.array(x) for x in [bxs,cfs,cls])
            dets.extend([([*map(float,b)],float(c),int(cl)) for b,c,cl in zip(bxs,cfs,cls) if int(cl)==0])
        trks = TRACKER.update_tracks(dets,frame=frm)
        for t in trks:
            if not t.is_confirmed(): continue
            tid = t.track_id
            x1,y1,x2,y2 = map(int,t.to_ltrb())
            cx,cy = int((x1+x2)/2),int((y1+y2)/2)
            zn = next((n for n,poly in ZONES.items() if poly.contains(Point(cx,cy))),None)
            pp = STATE["profiles"].get(tid)
            cloth = get_clothing(frm,(x1,y1,x2,y2)) if not pp or pp.get("clothing") in (None,"Unknown") else pp["clothing"]
            beh = get_behavior(tid,x1,y1,x2,y2)
            prof = update_prof(tid,cloth,beh)
            STATE["visits"].setdefault(tid,[])
            if zn and zn not in STATE["visits"][tid]: STATE["visits"][tid].append(zn)
            zst = zone_status(tid,zn) if zn else "Outside"
            pzn = STATE["last_zone"].get(tid)
            if zn!=pzn:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = f"[{ts}] ID {tid} "+("ENTERED "+zn if pzn is None and zn else "LEFT "+pzn if pzn and not zn else f"MOVED {pzn} -> {zn}")
                print(msg)
                STATE["events"].append({"ID":tid,"Clothing":prof["clothing"],"Behavior":beh,"Zone":zn or "Outside","Status":zst,"Visited":";".join(STATE["visits"].get(tid,[])),"Timestamp":ts})
                STATE["last_zone"][tid] = zn
            col = (0,0,200) if "Angry" in beh or "Agitated" in beh else (0,100,255) if "Anxious" in beh else (0,165,255) if "Confused" in beh else (128,0,128) if "Sad" in beh or "Tired" in beh else (0,255,0)
            thk = 4 if zst=="Unusual" else 2
            cv2.rectangle(frm,(x1,y1),(x2,y2),col,thk)
            y=y1-6
            for txt,clr,sz in [(f"ID:{tid}",col,0.6),(prof["clothing"],(255,165,0),0.5),(f"Beh:{beh}",(147,20,255),0.5),(f"Zone:{zn or 'Outside'}",(255,0,0),0.5)]:
                cv2.putText(frm,txt,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,sz,clr,2 if "ID" in txt else 1); y-=20 if "ID" in txt else 16
            cv2.putText(frm,f"Visited:{len(STATE['visits'].get(tid,[]))}/{len(ZONES)}",(x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),1)
        cv2.line(frm,(426,0),(426,CFG['h']),(255,255,0),2); cv2.line(frm,(853,0),(853,CFG['h']),(255,255,0),2)
        for txt,x in [("Reception",50),("Office",480),("Meeting Room",950)]: cv2.putText(frm,txt,(x,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)
        cv2.putText(frm,f"Tracking: {len(STATE['last_zone'])}",(10,CFG['h']-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.imshow("Office VLM",frm)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()
    if STATE["events"]: pd.DataFrame(STATE["events"]).to_csv(CFG['csv'],index=False); print(f"Saved {len(STATE['events'])} events")
    print("\nProfiles:")
    for pid,p in STATE["profiles"].items(): print(f"ID {pid}: {p['clothing']} | {', '.join(p['behaviors'])} | {', '.join(STATE['visits'].get(pid,[]))} | {p['first_seen']}")

if __name__=="__main__": main()