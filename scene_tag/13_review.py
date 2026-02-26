#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驾驶行为标注 审核+修改 工具

单页面 Web 界面，通过按钮切换审核/修改模式：
  审核模式：标记正确/错误/待定，统计 Precision（不可改标注）
  修改模式：可改标签、调时间、增删段，导出为训练数据

用法:
    python 13_review.py --port 9000
    浏览器打开后粘贴标注文件路径即可开始
"""

import os, sys, json, argparse, mimetypes, urllib.parse, threading, datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from collections import Counter

DEFAULT_PORT = 9000

ALL_LABELS = [
    "TrafficLight_StraightStopOrGo", "TrafficLight_LeftTurnStopOrGo",
    "TrafficLight_DarkIntersectionPass",
    "LaneChange_NavForIntersection", "LaneChange_AvoidSlowVRU",
    "LaneChange_AvoidStaticVehicle",
    "DynamicInteraction_VRUInLaneCrossing", "DynamicInteraction_VehicleInLaneCrossing",
    "DynamicInteraction_StandardVehicleCutIn", "DynamicInteraction_LeadVehicleEmergencyBrake",
    "AbnormalStop_StuckWaiting",
    "StartStop_StartFromMainRoad", "StartStop_ParkRoadside",
    "Intersection_LeftTurn", "Intersection_RightTurn", "Intersection_StandardUTurn",
    "LaneCruising_Straight", "else",
]

LABEL_COLORS = {
    "TrafficLight_StraightStopOrGo": "#e74c3c", "TrafficLight_LeftTurnStopOrGo": "#c0392b",
    "TrafficLight_DarkIntersectionPass": "#a93226",
    "LaneChange_NavForIntersection": "#3498db", "LaneChange_AvoidSlowVRU": "#2980b9",
    "LaneChange_AvoidStaticVehicle": "#2471a3",
    "DynamicInteraction_VRUInLaneCrossing": "#e67e22", "DynamicInteraction_VehicleInLaneCrossing": "#d35400",
    "DynamicInteraction_StandardVehicleCutIn": "#f39c12", "DynamicInteraction_LeadVehicleEmergencyBrake": "#ff0000",
    "AbnormalStop_StuckWaiting": "#d4ac0d",
    "StartStop_StartFromMainRoad": "#9b59b6", "StartStop_ParkRoadside": "#8e44ad",
    "Intersection_LeftTurn": "#1abc9c", "Intersection_RightTurn": "#16a085",
    "Intersection_StandardUTurn": "#27ae60", "LaneCruising_Straight": "#95a5a6", "else": "#7f8c8d",
}

LABEL_DEFINITIONS = {
    "TrafficLight_StraightStopOrGo": "在红绿灯前直行停车/起步（须有停→走或走→停转换）",
    "TrafficLight_LeftTurnStopOrGo": "在红绿灯前左转停车/起步（左转车道即适用）",
    "TrafficLight_DarkIntersectionPass": "红绿灯熄灭路口（困住/脱困）",
    "AbnormalStop_StuckWaiting": "异常停留（无保护左转/右转让行/会车等困住场景）",
    "LaneChange_NavForIntersection": "路口附近变道（从直行道变到转弯道等）",
    "LaneChange_AvoidSlowVRU": "变道避让慢行行人/骑行者",
    "LaneChange_AvoidStaticVehicle": "变道避让静止车辆",
    "DynamicInteraction_VRUInLaneCrossing": "行人/骑行者横穿车道",
    "DynamicInteraction_VehicleInLaneCrossing": "其他车辆横穿车道",
    "DynamicInteraction_StandardVehicleCutIn": "其他车辆加塞",
    "DynamicInteraction_LeadVehicleEmergencyBrake": "前车紧急刹车",
    "StartStop_StartFromMainRoad": "主路停车后起步（非红绿灯）",
    "StartStop_ParkRoadside": "靠边停车",
    "Intersection_LeftTurn": "路口左转",
    "Intersection_RightTurn": "路口右转",
    "Intersection_StandardUTurn": "路口掉头",
    "LaneCruising_Straight": "定速直行巡航（极严格：无任何交互）",
    "else": "不属于以上任何类别",
}


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>驾驶行为标注 审核+修改</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI',sans-serif;background:#1a1a2e;color:#eee}

.landing-overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(10,10,30,.95);z-index:1000;display:flex;align-items:center;justify-content:center}
.landing-overlay.hidden{display:none}
.landing-box{background:#16213e;padding:40px;border-radius:16px;width:560px;max-width:90vw;box-shadow:0 20px 60px rgba(0,0,0,.5)}
.landing-box h2{color:#e94560;margin-bottom:20px;font-size:22px}
.lf{margin-bottom:16px}
.lf label{display:block;font-size:13px;color:#aaa;margin-bottom:4px}
.lf input[type="text"]{width:100%;padding:10px 14px;background:#0f3460;color:#eee;border:1px solid #1a3a6e;border-radius:6px;font-size:14px}
.lf input:focus{outline:none;border-color:#e94560}
.load-btn{width:100%;padding:12px;background:#e94560;color:#fff;border:none;border-radius:8px;font-size:16px;font-weight:bold;cursor:pointer;margin-top:8px}
.load-btn:hover{background:#d63551}
.load-btn:disabled{background:#555;cursor:not-allowed}
.load-error{color:#ff6b6b;font-size:13px;margin-top:8px;min-height:18px}
.lhist{margin-top:10px;font-size:12px;color:#555}
.lhist select{width:100%;padding:5px;background:#0f3460;color:#aaa;border:1px solid #1a3a6e;border-radius:4px;font-size:12px;margin-top:3px}

.header{background:#16213e;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #0f3460}
.header h1{font-size:17px;color:#e94560}
.hdr-right{display:flex;align-items:center;gap:8px;font-size:12px;color:#aaa}
.hdr-right button{background:#0f3460;color:#aaa;border:1px solid #1a3a6e;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px}
.hdr-right button:hover{background:#1a3a6e;color:#eee}
.mode-toggle{display:inline-flex;border-radius:6px;overflow:hidden;border:1px solid #1a3a6e}
.mode-toggle button{padding:5px 14px;border:none;font-size:12px;font-weight:bold;cursor:pointer;transition:all .15s}
.mode-toggle .mt-review{background:#0f3460;color:#aaa}
.mode-toggle .mt-augment{background:#0f3460;color:#aaa}
.mode-toggle .mt-review.active{background:#27ae60;color:#fff}
.mode-toggle .mt-augment.active{background:#e67e22;color:#fff}

.controls{background:#16213e;padding:8px 20px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #0f3460}
.controls select,.controls input{background:#0f3460;color:#eee;border:1px solid #1a3a6e;padding:5px 8px;border-radius:4px;font-size:12px}
.controls select{min-width:180px}
.controls input[type="text"]{width:55px;text-align:center}
.controls label{font-size:12px;color:#888}
.controls button{background:#0f3460;color:#eee;border:1px solid #1a3a6e;padding:5px 10px;border-radius:4px;cursor:pointer;font-size:12px}
.controls button:hover{background:#1a3a6e}
.progress-bar{flex:1;min-width:120px;height:5px;background:#0f3460;border-radius:3px;overflow:hidden}
.progress-fill{height:100%;background:#e94560;transition:width .3s}

.main{display:flex;height:calc(100vh - 92px)}

.video-panel{flex:1;padding:12px;display:flex;flex-direction:column;overflow-y:auto}
video{width:100%;max-height:50vh;background:#000;border-radius:8px}
.timeline{margin-top:10px;position:relative;height:40px;background:#0f3460;border-radius:6px;overflow:visible;cursor:pointer}
.timeline .seg{position:absolute;top:0;height:100%;opacity:.85;display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5);overflow:hidden;white-space:nowrap;border-right:1px solid rgba(0,0,0,.3)}
.timeline .seg.reviewed-correct{border-bottom:3px solid #2ecc71}
.timeline .seg.reviewed-wrong{border-bottom:3px solid #ff6b6b}
.timeline .seg.reviewed-unsure{border-bottom:3px solid #f1c40f}
.timeline .seg.modified{border-top:3px solid #e67e22}
.timeline .playhead{position:absolute;top:0;width:2px;height:100%;background:#fff;pointer-events:none;z-index:10;transition:left .1s linear}
.time-labels{display:flex;justify-content:space-between;font-size:10px;color:#555;margin-top:2px}

/* Label reference grid */
.label-ref{margin-top:14px;padding:10px;background:#0d1b36;border-radius:8px;border:1px solid #0f3460}
.label-ref h4{font-size:12px;color:#e94560;margin-bottom:8px;cursor:pointer}
.label-grid{display:flex;flex-wrap:wrap;gap:4px}
.label-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:12px;font-size:10px;color:#ccc;background:#16213e;border:1px solid #1a3a6e;cursor:default;line-height:1.3}
.label-chip .lc-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.label-chip:hover{background:#1a3a6e}
.label-chip .lc-def{display:none;position:absolute;bottom:100%;left:0;background:#1a1a2e;color:#ccc;padding:6px 10px;border-radius:6px;font-size:11px;white-space:nowrap;z-index:100;border:1px solid #0f3460;pointer-events:none}
.label-chip:hover .lc-def{display:block}
.label-chip{position:relative}

.info-panel{width:400px;background:#16213e;padding:14px;overflow-y:auto;border-left:1px solid #0f3460}
.video-name{font-size:12px;color:#aaa;word-break:break-all;margin-bottom:10px;line-height:1.4}

.seg-list{margin-bottom:12px}
.seg-item{padding:8px 10px;margin-bottom:6px;background:#0f3460;border-radius:8px;font-size:12px;border-left:4px solid transparent;cursor:pointer;transition:all .12s}
.seg-item:hover{background:#132d5e}
.seg-item.selected{background:#1a3a6e;border-left-color:#e94560}
.seg-item.review-correct{border-left-color:#2ecc71}
.seg-item.review-wrong{border-left-color:#ff6b6b}
.seg-item.review-unsure{border-left-color:#f1c40f}
.seg-item.seg-modified{border-right:3px solid #e67e22}
.seg-header{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.seg-idx{background:#e94560;color:#fff;width:18px;height:18px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:bold;flex-shrink:0}
.seg-color{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.seg-label{flex:1;font-weight:500;font-size:11px}
.seg-time{color:#aaa;font-size:11px;flex-shrink:0}
.seg-conf{color:#e94560;font-size:11px;flex-shrink:0;min-width:32px;text-align:right}
.modified-badge{display:inline-block;background:#e67e22;color:#fff;font-size:9px;padding:1px 5px;border-radius:8px;margin-left:4px}

.seg-edit{margin-top:6px;padding:6px;background:#0a1a35;border-radius:6px}
.seg-edit-row{display:flex;align-items:center;gap:5px;font-size:11px;margin-bottom:4px}
.seg-edit-row:last-child{margin-bottom:0}
.seg-edit-row label{color:#888;min-width:32px;font-size:11px}
.seg-edit-row select{background:#0a2040;color:#eee;border:1px solid #1a3a6e;border-radius:4px;padding:3px 4px;font-size:11px;flex:1;cursor:pointer;-webkit-appearance:menulist;appearance:menulist}
.seg-edit-row input[type="number"]{background:#0a2040;color:#eee;border:1px solid #1a3a6e;border-radius:4px;padding:3px 4px;font-size:11px;width:48px;text-align:center}
.seg-edit-row .sav-btn{background:#27ae60;color:#fff;border:none;border-radius:4px;padding:3px 8px;cursor:pointer;font-size:10px;font-weight:bold}
.seg-edit-row .sav-btn:hover{background:#2ecc71}
.seg-edit-row .del-btn{background:#e74c3c;color:#fff;border:none;border-radius:4px;padding:3px 8px;cursor:pointer;font-size:10px;font-weight:bold}

.seg-review-btns{display:flex;gap:3px;margin-top:4px}
.seg-review-btns button{padding:3px 8px;border:none;border-radius:4px;font-size:10px;font-weight:bold;cursor:pointer;opacity:.7;transition:all .12s}
.seg-review-btns button:hover{opacity:1}
.seg-review-btns button.active{opacity:1;box-shadow:0 0 6px rgba(255,255,255,.2)}
.seg-btn-correct{background:#27ae60;color:#fff}
.seg-btn-correct.active{background:#2ecc71}
.seg-btn-unsure{background:#f39c12;color:#fff}
.seg-btn-unsure.active{background:#f1c40f}
.seg-btn-wrong{background:#e74c3c;color:#fff}
.seg-btn-wrong.active{background:#ff6b6b}

.add-seg-btn{width:100%;padding:6px;background:#1a3a6e;color:#aaa;border:1px dashed #3a5a8e;border-radius:6px;cursor:pointer;font-size:11px;text-align:center;display:none}
.add-seg-btn:hover{background:#2a4a7e;color:#eee}

.comment-box{margin-top:10px}
.comment-box textarea{width:100%;height:40px;background:#0f3460;color:#eee;border:1px solid #1a3a6e;border-radius:6px;padding:6px;font-size:12px;resize:vertical}

.nav-btns{display:flex;gap:6px;margin-top:10px}
.nav-btns button{flex:1;padding:8px;background:#0f3460;color:#eee;border:1px solid #1a3a6e;border-radius:6px;cursor:pointer;font-size:12px}
.nav-btns button:hover{background:#1a3a6e}

.batch-btns{display:flex;gap:4px;margin-top:6px}
.batch-btns button{flex:1;padding:5px;border:none;border-radius:4px;font-size:11px;cursor:pointer}

.summary{margin-top:12px;padding-top:10px;border-top:1px solid #0f3460}
.summary h3{font-size:13px;margin-bottom:6px;color:#e94560}
.stat-row{display:flex;justify-content:space-between;font-size:12px;padding:2px 0;color:#aaa}
.stat-row span:last-child{color:#eee}

.label-accuracy{margin-top:10px;padding-top:10px;border-top:1px solid #0f3460}
.label-accuracy h3{font-size:13px;margin-bottom:6px;color:#e94560;cursor:pointer}
.acc-row{display:flex;align-items:center;gap:5px;font-size:11px;padding:2px 0;color:#aaa}
.acc-bar{flex:1;height:3px;background:#0f3460;border-radius:2px;overflow:hidden}
.acc-fill{height:100%;border-radius:2px}
.acc-name{min-width:90px;font-size:10px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.acc-pct{min-width:40px;text-align:right;font-size:10px;color:#eee}

.shortcuts{margin-top:8px;font-size:10px;color:#555;line-height:1.8}
.kbd{display:inline-block;background:#0f3460;padding:1px 5px;border-radius:3px;font-size:10px;color:#aaa;border:1px solid #1a3a6e}
</style>
</head>
<body>

<div class="landing-overlay" id="landingOverlay">
<div class="landing-box">
  <h2>驾驶行为标注 审核+修改</h2>
  <div class="lf"><label>标注文件路径（粘贴服务器端路径）:</label>
    <input type="text" id="annoPathInput" placeholder="/mnt/pfs/houhaotian/annotations_junction_2.5w.json"/></div>
  <div class="lf"><label>审核结果保存路径（可选，留空自动生成）:</label>
    <input type="text" id="reviewPathInput" placeholder="留空则自动生成"/></div>
  <button class="load-btn" id="loadBtn" onclick="loadAnnotations()">加载数据</button>
  <div id="loadError" class="load-error"></div>
  <div class="lhist" id="loadHistory" style="display:none"><label>最近:</label>
    <select id="historySelect" onchange="document.getElementById('annoPathInput').value=this.value"></select></div>
</div></div>

<div class="header">
  <div style="display:flex;align-items:center;gap:12px">
    <h1>标注审核</h1>
    <div class="mode-toggle">
      <button class="mt-review active" onclick="switchMode('review')">审核</button>
      <button class="mt-augment" onclick="switchMode('augment')">修改</button>
    </div>
  </div>
  <div class="hdr-right">
    <button onclick="showLanding()">切换文件</button>
    <button onclick="reloadData()">刷新</button>
    <button onclick="exportCSV()">导出CSV</button>
    <button id="exportTrainBtn" onclick="exportTraining()" style="background:#e67e22;color:#fff;display:none">导出训练数据</button>
    <span id="headerStatsText"></span>
  </div>
</div>

<div class="controls">
  <label>标签:</label><select id="filterLabel"></select>
  <label>状态:</label><select id="filterStatus"></select>
  <label>跳转:</label><input type="text" id="jumpInput" placeholder="#"/>
  <button onclick="jumpTo()">GO</button>
  <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
</div>

<div class="main">
  <div class="video-panel">
    <video id="videoPlayer" controls preload="metadata"></video>
    <div class="timeline" id="timeline" onclick="seekTimeline(event)"></div>
    <div class="time-labels"><span>0s</span><span>5s</span><span>10s</span><span>15s</span><span>20s</span></div>
    <div class="label-ref" id="labelRef">
      <h4 onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'flex':'none'">可用标签参考 (点击折叠/展开)</h4>
      <div class="label-grid" id="labelGrid"></div>
    </div>
  </div>

  <div class="info-panel">
    <div class="video-name" id="videoName"></div>
    <div class="seg-list" id="segList"></div>
    <button class="add-seg-btn" id="addSegBtn" onclick="addSegment()">+ 添加标注段</button>

    <div class="batch-btns">
      <button style="background:#27ae60;color:#fff" onclick="markAllSegs('correct')">全部正确</button>
      <button style="background:#f39c12;color:#fff" onclick="markAllSegs('unsure')">全部待定</button>
      <button style="background:#e74c3c;color:#fff" onclick="markAllSegs('wrong')">全部错误</button>
    </div>
    <div class="comment-box"><textarea id="commentBox" placeholder="备注..." onchange="saveComment()"></textarea></div>
    <div class="nav-btns">
      <button onclick="navigate(-1)">&#9664; 上一个</button>
      <button onclick="navigate(1)">下一个 &#9654;</button>
    </div>

    <div class="summary">
      <h3>审核进度</h3>
      <div class="stat-row"><span>总段数</span><span id="sTotal">-</span></div>
      <div class="stat-row"><span>已审核</span><span id="sReviewed">-</span></div>
      <div class="stat-row"><span style="color:#2ecc71">正确</span><span id="sCorrect">-</span></div>
      <div class="stat-row"><span style="color:#ff6b6b">错误</span><span id="sWrong">-</span></div>
      <div class="stat-row"><span style="color:#f1c40f">待定</span><span id="sUnsure">-</span></div>
      <div class="stat-row" id="sModifiedRow" style="display:none"><span style="color:#e67e22">已修改</span><span id="sModified">-</span></div>
    </div>
    <div class="label-accuracy"><h3 onclick="toggleAccuracy()">各标签准确率 ▼</h3><div id="accList"></div></div>
    <div class="shortcuts">
      <span class="kbd">1-9</span> 选段 <span class="kbd">A</span> 正确 <span class="kbd">S</span> 待定 <span class="kbd">D</span> 错误 <span class="kbd">Q</span> 全正确 <span class="kbd">←→</span> 切换 <span class="kbd">Space</span> 播放 <span class="kbd">M</span> 切模式
    </div>
  </div>
</div>

<script>
let annotations=[], filteredIndices=[], currentFilteredPos=0, reviews={};
let selectedSegIdx=0, accVisible=true, currentVideoPath=null;
let currentMode='review', loadedAnnoPath='', loadedReviewPath='';
let _saveTimer=null;
const COLORS=LABEL_COLORS_PLACEHOLDER;
const ALL_LABELS=ALL_LABELS_PLACEHOLDER;
const LABEL_DEFS=LABEL_DEFS_PLACEHOLDER;

// ===== Mode =====
function switchMode(m){
  currentMode=m;
  document.querySelector('.mt-review').classList.toggle('active',m==='review');
  document.querySelector('.mt-augment').classList.toggle('active',m==='augment');
  document.getElementById('exportTrainBtn').style.display=m==='augment'?'':'none';
  document.getElementById('sModifiedRow').style.display=m==='augment'?'':'none';
  document.getElementById('addSegBtn').style.display=m==='augment'?'block':'none';
  render();
}

// ===== Landing =====
function showLanding(){document.getElementById('landingOverlay').classList.remove('hidden')}
function initHistory(){
  const h=JSON.parse(localStorage.getItem('rvHist')||'[]');
  if(!h.length)return;
  document.getElementById('loadHistory').style.display='block';
  const s=document.getElementById('historySelect');
  s.innerHTML='<option value="">--</option>';
  h.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent=p.split('/').pop();s.appendChild(o)});
  if(!document.getElementById('annoPathInput').value)document.getElementById('annoPathInput').value=h[0];
}
function addHistory(p){
  let h=JSON.parse(localStorage.getItem('rvHist')||'[]');
  h=h.filter(x=>x!==p);h.unshift(p);h=h.slice(0,10);
  localStorage.setItem('rvHist',JSON.stringify(h));
}

async function loadAnnotations(){
  const path=document.getElementById('annoPathInput').value.trim();
  const rp=document.getElementById('reviewPathInput').value.trim();
  const err=document.getElementById('loadError');
  if(!path){err.textContent='请输入路径';return}
  const btn=document.getElementById('loadBtn');
  btn.disabled=true;btn.textContent='加载中...';err.textContent='';
  try{
    const r=await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,review_path:rp,mode:currentMode})});
    const d=await r.json();
    if(d.error){err.textContent=d.error;return}
    annotations=d.annotations;reviews=d.reviews||{};
    loadedAnnoPath=path;loadedReviewPath=d.review_path||'';
    addHistory(path);
    document.getElementById('landingOverlay').classList.add('hidden');
    buildFilters();applyFilter();render();
  }catch(e){err.textContent='加载失败: '+e.message}
  finally{btn.disabled=false;btn.textContent='加载数据'}
}

async function reloadData(){
  if(!loadedAnnoPath)return;
  try{
    const r=await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:loadedAnnoPath,review_path:loadedReviewPath,mode:currentMode})});
    const d=await r.json();if(d.error){alert(d.error);return}
    const old=annotations.length;annotations=d.annotations;
    buildFilters();applyFilter();render();
    alert(old+' -> '+annotations.length+' 个视频');
  }catch(e){alert(e.message)}
}

// ===== Effective segments (original + added - deleted) =====
function getEffectiveSegs(vp, origSegs){
  const rv=reviews[vp]||{};
  const del=rv.deleted_segments||[];
  const added=rv.added_segments||[];
  const mods=rv.modifications||{};
  let segs=[];
  origSegs.forEach((s,i)=>{
    if(del.includes(i))return;
    const m=mods[String(i)];
    segs.push({...s, _origIdx:i, _isAdded:false, label:m?m.label:s.label, start:m?m.start:s.start, end:m?m.end:s.end, _modified:!!m});
  });
  added.forEach((s,i)=>{
    segs.push({...s, _origIdx:'a'+i, _isAdded:true, _modified:false, confidence:100});
  });
  return segs;
}

// ===== Review helpers =====
function getSegReview(vp,key){const r=reviews[vp];return r&&r.segments?r.segments[String(key)]||null:null}
function setSegReview(vp,key,status){
  if(!reviews[vp])reviews[vp]={segments:{},comment:'',modifications:{},added_segments:[],deleted_segments:[]};
  if(!reviews[vp].segments)reviews[vp].segments={};
  reviews[vp].segments[String(key)]=status;debounceSave();
}
function getSegMod(vp,origIdx){const r=reviews[vp];return r&&r.modifications?r.modifications[String(origIdx)]||null:null}

function debounceSave(){clearTimeout(_saveTimer);_saveTimer=setTimeout(()=>saveReviews(),500)}
async function saveReviews(){
  if(!loadedReviewPath)return;
  try{await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({review_path:loadedReviewPath,data:reviews})})}catch(e){}
}

// ===== Filters (dual independent) =====
function buildFilters(){
  // Label filter
  const fl=document.getElementById('filterLabel');
  const curL=fl.value;
  fl.innerHTML='<option value="all">全部标签</option>';
  const labels=new Set();
  annotations.forEach(a=>(a.segments||[]).forEach(s=>labels.add(s.label)));
  [...labels].sort().forEach(l=>{const o=document.createElement('option');o.value=l;o.textContent=l;fl.appendChild(o)});
  if([...fl.options].some(o=>o.value===curL))fl.value=curL;

  // Status filter
  const fs=document.getElementById('filterStatus');
  const curS=fs.value;
  fs.innerHTML='<option value="all">全部状态</option><option value="unreviewed">未审核</option><option value="correct">已标正确</option><option value="wrong">已标错误</option><option value="unsure">已标待定</option><option value="modified">已修改</option>';
  if(curS)fs.value=curS;

  fl.onchange=fs.onchange=()=>{applyFilter();currentVideoPath=null;render()};
}

function applyFilter(){
  const lv=document.getElementById('filterLabel').value;
  const sv=document.getElementById('filterStatus').value;
  filteredIndices=[];
  for(let i=0;i<annotations.length;i++){
    const a=annotations[i], vp=a.video_path, segs=a.segments||[];
    const rv=reviews[vp]||{};
    // Label filter
    if(lv!=='all'&&!segs.some(s=>s.label===lv))continue;
    // Status filter
    if(sv!=='all'){
      const segRv=rv.segments||{};
      const mods=rv.modifications||{};
      let match=false;
      if(sv==='unreviewed') match=segs.some((_,si)=>!segRv[String(si)]);
      else if(sv==='correct') match=segs.length>0&&segs.every((_,si)=>segRv[String(si)]==='correct');
      else if(sv==='wrong') match=Object.values(segRv).some(s=>s==='wrong');
      else if(sv==='unsure') match=Object.values(segRv).some(s=>s==='unsure');
      else if(sv==='modified') match=Object.keys(mods).length>0||(rv.added_segments||[]).length>0;
      if(!match)continue;
    }
    filteredIndices.push(i);
  }
  currentFilteredPos=Math.min(currentFilteredPos,Math.max(0,filteredIndices.length-1));
}

// ===== Render =====
function render(){
  if(!filteredIndices.length){
    document.getElementById('videoName').textContent='没有匹配的视频';
    document.getElementById('segList').innerHTML='';
    document.getElementById('timeline').innerHTML='';
    updateStats();return;
  }
  const idx=filteredIndices[currentFilteredPos];
  const a=annotations[idx], vp=a.video_path;
  const eSegs=getEffectiveSegs(vp,a.segments||[]);

  const video=document.getElementById('videoPlayer');
  if(currentVideoPath!==vp){currentVideoPath=vp;video.src='/video/'+encodeURIComponent(vp);video.load()}

  const parts=vp.split('/');
  const rvd=eSegs.filter(s=>getSegReview(vp,s._origIdx)).length;
  document.getElementById('videoName').innerHTML=
    '<strong>#'+(idx+1)+'/'+annotations.length+'</strong> (筛选'+(currentFilteredPos+1)+'/'+filteredIndices.length+')<br>'+
    parts.slice(-5).join('/')+'<br>'+
    '<span style="color:'+(rvd===eSegs.length?'#2ecc71':'#e94560')+'">'+rvd+'/'+eSegs.length+' 段已审</span>';

  if(selectedSegIdx>=eSegs.length)selectedSegIdx=0;
  renderTimeline(eSegs,vp);
  renderSegList(eSegs,vp,video,a.segments||[]);

  const rv=reviews[vp];
  document.getElementById('commentBox').value=(rv&&rv.comment)||'';
  document.getElementById('addSegBtn').style.display=currentMode==='augment'?'block':'none';
  updateStats();
  document.getElementById('headerStatsText').textContent=
    filteredIndices.length+' 个 | #'+(currentFilteredPos+1)+' | '+(loadedAnnoPath.split('/').pop()||'');
  document.getElementById('jumpInput').value=idx+1;
}

function renderTimeline(eSegs,vp){
  const tl=document.getElementById('timeline');
  tl.innerHTML='<div class="playhead" id="playhead"></div>';
  eSegs.forEach((s,si)=>{
    const div=document.createElement('div');div.className='seg';
    const st=getSegReview(vp,s._origIdx);
    if(st)div.classList.add('reviewed-'+st);
    if(s._modified||s._isAdded)div.classList.add('modified');
    div.style.left=(s.start/20*100)+'%';
    div.style.width=Math.max((s.end-s.start)/20*100,1)+'%';
    div.style.background=COLORS[s.label]||'#555';
    div.textContent=s.label.split('_').pop();
    div.onclick=e=>{e.stopPropagation();selectedSegIdx=si;render()};
    tl.appendChild(div);
  });
}

function renderSegList(eSegs,vp,video,origSegs){
  const sl=document.getElementById('segList');
  sl.innerHTML='';
  eSegs.forEach((s,si)=>{
    const key=s._origIdx;
    const st=getSegReview(vp,key);
    const div=document.createElement('div');
    div.className='seg-item'+(si===selectedSegIdx?' selected':'');
    if(st)div.classList.add('review-'+st);
    if(s._modified||s._isAdded)div.classList.add('seg-modified');

    let html='<div class="seg-header">'+
      '<div class="seg-idx">'+(si+1)+'</div>'+
      '<div class="seg-color" style="background:'+(COLORS[s.label]||'#555')+'"></div>'+
      '<div class="seg-label">'+s.label+(s._modified?'<span class="modified-badge">改</span>':'')+(s._isAdded?'<span class="modified-badge" style="background:#3498db">新</span>':'')+'</div>'+
      '<div class="seg-time">'+s.start+'s-'+s.end+'s</div>'+
      '<div class="seg-conf">'+s.confidence+'%</div></div>';

    // Edit area in augment mode for selected segment
    if(currentMode==='augment'&&si===selectedSegIdx){
      let opts='';
      ALL_LABELS.forEach(l=>{opts+='<option value="'+l+'"'+(l===s.label?' selected':'')+'>'+l+'</option>'});
      html+='<div class="seg-edit" onclick="event.stopPropagation()">'+
        '<div class="seg-edit-row"><label>标签:</label><select id="eL_'+si+'" onmousedown="event.stopPropagation()">'+opts+'</select></div>'+
        '<div class="seg-edit-row"><label>时间:</label>'+
        '<input type="number" id="eS_'+si+'" value="'+s.start+'" min="0" max="20" step="1" onclick="event.stopPropagation()"/>'+
        '<span style="color:#666">-</span>'+
        '<input type="number" id="eE_'+si+'" value="'+s.end+'" min="0" max="20" step="1" onclick="event.stopPropagation()"/>'+
        '<span style="color:#555;font-size:10px">秒</span>'+
        '<button class="sav-btn" onclick="event.stopPropagation();saveEdit('+si+')">保存</button>'+
        '<button class="del-btn" onclick="event.stopPropagation();deleteSeg('+si+')">删除</button></div>';
      if(s._modified&&!s._isAdded){
        const orig=origSegs[s._origIdx];
        if(orig)html+='<div style="font-size:10px;color:#666;margin-top:2px">原: '+orig.label+' ['+orig.start+'s-'+orig.end+'s]</div>';
      }
      html+='</div>';
    }

    html+='<div class="seg-review-btns">'+
      '<button class="seg-btn-correct'+(st==='correct'?' active':'')+'" onclick="event.stopPropagation();markSeg('+si+',\'correct\')">✓正确</button>'+
      '<button class="seg-btn-unsure'+(st==='unsure'?' active':'')+'" onclick="event.stopPropagation();markSeg('+si+',\'unsure\')">?待定</button>'+
      '<button class="seg-btn-wrong'+(st==='wrong'?' active':'')+'" onclick="event.stopPropagation();markSeg('+si+',\'wrong\')">✗错误</button></div>';

    div.innerHTML=html;
    div.onclick=()=>{
      if(selectedSegIdx===si)return; // don't re-render if already selected (fixes dropdown bug)
      selectedSegIdx=si;video.currentTime=s.start;video.play();render();
    };
    sl.appendChild(div);
  });
}

// ===== Edit actions =====
function saveEdit(si){
  const idx=filteredIndices[currentFilteredPos];
  const a=annotations[idx], vp=a.video_path;
  const eSegs=getEffectiveSegs(vp,a.segments||[]);
  const seg=eSegs[si];
  const nl=document.getElementById('eL_'+si).value;
  const ns=parseInt(document.getElementById('eS_'+si).value);
  const ne=parseInt(document.getElementById('eE_'+si).value);
  if(isNaN(ns)||isNaN(ne)||ns>=ne||ns<0||ne>20){alert('时间无效');return}

  if(!reviews[vp])reviews[vp]={segments:{},comment:'',modifications:{},added_segments:[],deleted_segments:[]};

  if(seg._isAdded){
    const ai=parseInt(String(seg._origIdx).substring(1));
    if(!reviews[vp].added_segments)reviews[vp].added_segments=[];
    reviews[vp].added_segments[ai]={label:nl,start:ns,end:ne};
  }else{
    if(!reviews[vp].modifications)reviews[vp].modifications={};
    reviews[vp].modifications[String(seg._origIdx)]={label:nl,start:ns,end:ne};
  }
  if(!reviews[vp].segments)reviews[vp].segments={};
  reviews[vp].segments[String(seg._origIdx)]='correct';
  debounceSave();render();
}

function deleteSeg(si){
  const idx=filteredIndices[currentFilteredPos];
  const a=annotations[idx], vp=a.video_path;
  const eSegs=getEffectiveSegs(vp,a.segments||[]);
  const seg=eSegs[si];
  if(!reviews[vp])reviews[vp]={segments:{},comment:'',modifications:{},added_segments:[],deleted_segments:[]};

  if(seg._isAdded){
    const ai=parseInt(String(seg._origIdx).substring(1));
    reviews[vp].added_segments.splice(ai,1);
    // Clean up review for this added segment
    delete reviews[vp].segments[String(seg._origIdx)];
  }else{
    if(!reviews[vp].deleted_segments)reviews[vp].deleted_segments=[];
    if(!reviews[vp].deleted_segments.includes(seg._origIdx))reviews[vp].deleted_segments.push(seg._origIdx);
    delete reviews[vp].segments[String(seg._origIdx)];
    if(reviews[vp].modifications)delete reviews[vp].modifications[String(seg._origIdx)];
  }
  selectedSegIdx=Math.max(0,si-1);
  debounceSave();render();
}

function addSegment(){
  const idx=filteredIndices[currentFilteredPos];
  if(idx===undefined)return;
  const vp=annotations[idx].video_path;
  if(!reviews[vp])reviews[vp]={segments:{},comment:'',modifications:{},added_segments:[],deleted_segments:[]};
  if(!reviews[vp].added_segments)reviews[vp].added_segments=[];
  reviews[vp].added_segments.push({label:'else',start:0,end:20});
  const eSegs=getEffectiveSegs(vp,annotations[idx].segments||[]);
  selectedSegIdx=eSegs.length-1;
  // Auto-mark added as correct
  const key='a'+(reviews[vp].added_segments.length-1);
  if(!reviews[vp].segments)reviews[vp].segments={};
  reviews[vp].segments[key]='correct';
  debounceSave();render();
}

function markSeg(si,status){
  const idx=filteredIndices[currentFilteredPos];
  const a=annotations[idx], vp=a.video_path;
  const eSegs=getEffectiveSegs(vp,a.segments||[]);
  setSegReview(vp,eSegs[si]._origIdx,status);
  let next=-1;
  for(let i=si+1;i<eSegs.length;i++){if(!getSegReview(vp,eSegs[i]._origIdx)){next=i;break}}
  selectedSegIdx=next>=0?next:si;
  render();
}

function markAllSegs(status){
  const idx=filteredIndices[currentFilteredPos];
  const a=annotations[idx], vp=a.video_path;
  getEffectiveSegs(vp,a.segments||[]).forEach(s=>setSegReview(vp,s._origIdx,status));
  render();setTimeout(()=>navigate(1),300);
}

// ===== Stats =====
function updateStats(){
  let tot=0,rvd=0,cor=0,wrg=0,uns=0,mod=0;
  annotations.forEach(a=>{
    const vp=a.video_path;
    getEffectiveSegs(vp,a.segments||[]).forEach(s=>{
      tot++;const st=getSegReview(vp,s._origIdx);
      if(st){rvd++;if(st==='correct')cor++;else if(st==='wrong')wrg++;else if(st==='unsure')uns++}
      if(s._modified||s._isAdded)mod++;
    });
  });
  document.getElementById('sTotal').textContent=tot;
  document.getElementById('sReviewed').textContent=rvd+' ('+(tot>0?(rvd/tot*100).toFixed(1):'0')+'%)';
  document.getElementById('sCorrect').textContent=cor;
  document.getElementById('sWrong').textContent=wrg;
  document.getElementById('sUnsure').textContent=uns;
  document.getElementById('sModified').textContent=mod;
  document.getElementById('progressFill').style.width=(tot>0?rvd/tot*100:0)+'%';
  renderAccuracy();
}

function renderAccuracy(){
  const ls={};
  annotations.forEach(a=>{
    const vp=a.video_path;
    getEffectiveSegs(vp,a.segments||[]).forEach(s=>{
      if(!ls[s.label])ls[s.label]={total:0,reviewed:0,correct:0,wrong:0,unsure:0};
      ls[s.label].total++;
      const st=getSegReview(vp,s._origIdx);
      if(st){ls[s.label].reviewed++;ls[s.label][st]++}
    });
  });
  const ad=document.getElementById('accList');
  if(!accVisible){ad.innerHTML='';return}
  ad.innerHTML='';
  let totalEst=0,totalAll=0;
  Object.entries(ls).sort((a,b)=>b[1].total-a[1].total).forEach(([l,st])=>{
    const p=st.reviewed>0?st.correct/st.reviewed*100:-1;
    const est=p>=0?Math.round(st.total*p/100):-1;
    if(est>=0){totalEst+=est;totalAll+=st.total}
    const c=p<0?'#555':p>=80?'#2ecc71':p>=50?'#f1c40f':'#ff6b6b';
    const r=document.createElement('div');r.className='acc-row';
    r.innerHTML='<div class="acc-name" title="'+l+'">'+l.split('_').slice(1).join('_')+'</div>'+
      '<div class="acc-bar"><div class="acc-fill" style="width:'+(p>=0?p:0)+'%;background:'+c+'"></div></div>'+
      '<div class="acc-pct">'+(p>=0?p.toFixed(0)+'%':'-')+'</div>'+
      '<div style="min-width:55px;font-size:10px;color:#666">'+st.correct+'/'+st.reviewed+'/'+st.total+'</div>'+
      '<div style="min-width:45px;font-size:10px;color:#3498db;text-align:right;font-weight:bold" title="预估可挖掘正确数=总数×准确率">'+(est>=0?est:'-')+'</div>';
    ad.appendChild(r);
  });
  // Total summary row
  const tr=document.createElement('div');tr.className='acc-row';
  tr.style.cssText='border-top:1px solid #1a3a6e;margin-top:4px;padding-top:4px';
  tr.innerHTML='<div class="acc-name" style="font-weight:bold;color:#e94560">合计</div>'+
    '<div class="acc-bar"></div><div class="acc-pct"></div>'+
    '<div style="min-width:55px;font-size:10px;color:#888">总'+totalAll+'</div>'+
    '<div style="min-width:45px;font-size:10px;color:#3498db;text-align:right;font-weight:bold" title="预估可挖掘正确总数">'+totalEst+'</div>';
  ad.appendChild(tr);
}
function toggleAccuracy(){accVisible=!accVisible;renderAccuracy()}

// ===== Nav =====
function saveComment(){
  const idx=filteredIndices[currentFilteredPos];if(idx===undefined)return;
  const vp=annotations[idx].video_path;
  if(!reviews[vp])reviews[vp]={segments:{},comment:'',modifications:{},added_segments:[],deleted_segments:[]};
  reviews[vp].comment=document.getElementById('commentBox').value;debounceSave();
}
function navigate(dir){
  const o=currentFilteredPos;
  currentFilteredPos=Math.max(0,Math.min(filteredIndices.length-1,currentFilteredPos+dir));
  if(currentFilteredPos!==o){selectedSegIdx=0;currentVideoPath=null}render();
}
function jumpTo(){
  const v=parseInt(document.getElementById('jumpInput').value);if(isNaN(v))return;
  const t=v-1;const p=filteredIndices.indexOf(t);
  if(p>=0)currentFilteredPos=p;
  else{currentFilteredPos=0;for(let i=0;i<filteredIndices.length;i++){if(filteredIndices[i]>=t){currentFilteredPos=i;break}}}
  selectedSegIdx=0;currentVideoPath=null;render();
}
function seekTimeline(e){
  const r=e.currentTarget.getBoundingClientRect();
  document.getElementById('videoPlayer').currentTime=(e.clientX-r.left)/r.width*20;
}

// ===== Export =====
function exportCSV(){
  let csv='video_path,seg_idx,label,start,end,confidence,review_status\n';
  annotations.forEach(a=>{
    const vp=a.video_path;
    getEffectiveSegs(vp,a.segments||[]).forEach((s,si)=>{
      csv+='"'+vp+'",'+si+',"'+s.label+'",'+s.start+','+s.end+','+s.confidence+','+(getSegReview(vp,s._origIdx)||'')+'\n';
    });
  });
  const b=new Blob([csv],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='review.csv';a.click();
}
async function exportTraining(){
  if(!loadedAnnoPath){alert('请先加载标注文件');return}
  try{
    const r=await fetch('/api/export_training',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({anno_path:loadedAnnoPath,reviews})});
    const d=await r.json();if(d.error){alert(d.error);return}
    let s='导出成功!\n\n目录: '+d.output_dir+'\n\n';
    Object.entries(d.stats).sort((a,b)=>b[1]-a[1]).forEach(([l,c])=>{s+='  '+l+': '+c+'条\n'});
    s+='\n共 '+d.total+' 条';alert(s);
  }catch(e){alert(e.message)}
}

// ===== Label reference grid =====
function buildLabelGrid(){
  const g=document.getElementById('labelGrid');g.innerHTML='';
  ALL_LABELS.forEach(l=>{
    const c=document.createElement('span');c.className='label-chip';
    c.innerHTML='<span class="lc-dot" style="background:'+(COLORS[l]||'#555')+'"></span>'+
      l.split('_').slice(1).join(' ')+'<span class="lc-def">'+(LABEL_DEFS[l]||l)+'</span>';
    g.appendChild(c);
  });
}

// Playhead
setInterval(()=>{const v=document.getElementById('videoPlayer'),p=document.getElementById('playhead');if(v&&p&&v.duration)p.style.left=(v.currentTime/20*100)+'%'},100);

// Keyboard
document.addEventListener('keydown',e=>{
  if(!document.getElementById('landingOverlay').classList.contains('hidden'))return;
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  const idx=filteredIndices[currentFilteredPos];if(idx===undefined)return;
  const eSegs=getEffectiveSegs(annotations[idx].video_path,annotations[idx].segments||[]);
  switch(e.key){
    case '1':case '2':case '3':case '4':case '5':case '6':case '7':case '8':case '9':{
      const si=parseInt(e.key)-1;
      if(si<eSegs.length){selectedSegIdx=si;const v=document.getElementById('videoPlayer');v.currentTime=eSegs[si].start;v.play();render()}break}
    case 'a':case 'A':markSeg(selectedSegIdx,'correct');break;
    case 's':case 'S':e.preventDefault();markSeg(selectedSegIdx,'unsure');break;
    case 'd':case 'D':markSeg(selectedSegIdx,'wrong');break;
    case 'q':case 'Q':markAllSegs('correct');break;
    case 'm':case 'M':switchMode(currentMode==='review'?'augment':'review');break;
    case 'ArrowLeft':navigate(-1);break;
    case 'ArrowRight':navigate(1);break;
    case ' ':e.preventDefault();const v=document.getElementById('videoPlayer');v.paused?v.play():v.pause();break;
  }
});

initHistory();buildLabelGrid();
</script>
</body></html>"""


# ==================== HTTP 服务 ====================
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

_save_lock = threading.Lock()

class ReviewHandler(BaseHTTPRequestHandler):
    # 无状态设计：不保存共享数据，每个客户端独立维护自己的 annotations/reviews
    rbufsize = -1
    wbufsize = 256 * 1024

    def log_message(self, format, *args): pass

    def do_GET(self):
        try:
            p = urllib.parse.urlparse(self.path).path
            if p in ("", "/"): self._serve_html()
            elif p.startswith("/video/"): self._serve_video(p)
            else: self.send_error(404)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as e: print(f"[ERR] GET {self.path}: {e}", file=sys.stderr)

    def do_POST(self):
        try:
            if self.path == "/api/save": self._save_reviews()
            elif self.path == "/api/load": self._handle_load()
            elif self.path == "/api/export_training": self._handle_export_training()
            else: self.send_error(404)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as e: print(f"[ERR] POST {self.path}: {e}", file=sys.stderr)

    def _serve_html(self):
        html = HTML_PAGE.replace("LABEL_COLORS_PLACEHOLDER", json.dumps(LABEL_COLORS))
        html = html.replace("ALL_LABELS_PLACEHOLDER", json.dumps(ALL_LABELS))
        html = html.replace("LABEL_DEFS_PLACEHOLDER", json.dumps(LABEL_DEFINITIONS, ensure_ascii=False))
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, path):
        vp = urllib.parse.unquote(path[7:])
        if not os.path.isfile(vp):
            self.send_error(404, f"Not found: {vp}"); return
        fs = os.path.getsize(vp)
        mt = mimetypes.guess_type(vp)[0] or "video/mp4"
        CS = 512 * 1024
        rh = self.headers.get("Range")
        if rh:
            rs = rh.replace("bytes=", "").split("-")
            s = int(rs[0]) if rs[0] else 0
            e = int(rs[1]) if rs[1] else fs - 1
            e = min(e, fs - 1); l = e - s + 1
            self.send_response(206)
            self.send_header("Content-Type", mt)
            self.send_header("Content-Length", str(l))
            self.send_header("Content-Range", f"bytes {s}-{e}/{fs}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(vp, "rb") as f:
                f.seek(s); rem = l
                while rem > 0:
                    c = f.read(min(CS, rem))
                    if not c: break
                    self.wfile.write(c); rem -= len(c)
        else:
            self.send_response(200)
            self.send_header("Content-Type", mt)
            self.send_header("Content-Length", str(fs))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(vp, "rb") as f:
                while True:
                    c = f.read(CS)
                    if not c: break
                    self.wfile.write(c)

    def _handle_load(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        ap = body.get("path", "")
        rp = body.get("review_path", "")
        if not ap: self._jr({"error": "请提供路径"}); return
        if not os.path.isfile(ap): self._jr({"error": f"不存在: {ap}"}); return
        try:
            with open(ap, "r", encoding="utf-8") as f: anns = json.load(f)
            if not isinstance(anns, list): self._jr({"error": "需要 JSON 数组"}); return
            if not rp: rp = os.path.splitext(ap)[0] + "_review.json"
            rvs = {}
            if os.path.exists(rp):
                try:
                    with open(rp, "r", encoding="utf-8") as f: rvs = json.load(f)
                    print(f"[载入] 审核 {len(rvs)} 条 ({rp})")
                except: rvs = {}
            od = os.path.dirname(rp)
            if od: os.makedirs(od, exist_ok=True)
            lc = Counter()
            for a in anns:
                for s in a.get("segments", []): lc[s["label"]] += 1
            print(f"[载入] {ap} ({len(anns)} 视频, {sum(lc.values())} 段)")
            self._jr({"annotations": anns, "reviews": rvs, "review_path": rp, "count": len(anns)})
        except json.JSONDecodeError as e: self._jr({"error": f"JSON 错误: {e}"})
        except Exception as e: self._jr({"error": f"失败: {e}"})

    def _save_reviews(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        rp = body.get("review_path", "")
        data = body.get("data", {})
        if not rp:
            self._jr({"error": "缺少 review_path"}); return
        try:
            od = os.path.dirname(rp)
            if od: os.makedirs(od, exist_ok=True)
            with _save_lock:
                with open(rp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            self._jr({"ok": True})
        except Exception as e:
            self._jr({"error": f"保存失败: {e}"})

    def _handle_export_training(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        crvs = body.get("reviews", {})
        anno_path = body.get("anno_path", "")
        if not anno_path or not os.path.isfile(anno_path):
            self._jr({"error": "标注文件不存在"}); return
        try:
            with open(anno_path, "r", encoding="utf-8") as f: annotations_data = json.load(f)
        except Exception as e:
            self._jr({"error": f"读取标注失败: {e}"}); return
        od = os.path.join(os.path.dirname(anno_path) or "results", "training_export")
        os.makedirs(od, exist_ok=True)
        by_label = {}; total = 0
        for ann in annotations_data:
            vp = ann.get("video_path", "")
            rv = crvs.get(vp, {})
            srvs = rv.get("segments", {}) if isinstance(rv, dict) else {}
            mods = rv.get("modifications", {}) if isinstance(rv, dict) else {}
            dels = rv.get("deleted_segments", []) if isinstance(rv, dict) else []
            added = rv.get("added_segments", []) if isinstance(rv, dict) else []
            # Original segments
            for si, seg in enumerate(ann.get("segments", [])):
                if si in dels: continue
                if srvs.get(str(si)) != "correct": continue
                m = mods.get(str(si))
                lb = m["label"] if m else seg["label"]
                st = m["start"] if m else seg["start"]
                en = m["end"] if m else seg["end"]
                by_label.setdefault(lb, []).append({"video_path": vp, "label": lb, "start": st, "end": en,
                    "confidence": seg.get("confidence", 0), "was_modified": m is not None})
                total += 1
            # Added segments
            for ai, aseg in enumerate(added):
                key = f"a{ai}"
                if srvs.get(key) != "correct": continue
                lb = aseg["label"]
                by_label.setdefault(lb, []).append({"video_path": vp, "label": lb, "start": aseg["start"],
                    "end": aseg["end"], "confidence": 100, "was_modified": False, "was_added": True})
                total += 1
        stats = {}
        with _save_lock:
            for lb, items in by_label.items():
                with open(os.path.join(od, f"{lb}.json"), "w", encoding="utf-8") as f:
                    json.dump(items, f, indent=2, ensure_ascii=False)
                stats[lb] = len(items)
            all_items = [i for items in by_label.values() for i in items]
            with open(os.path.join(od, "_all.json"), "w", encoding="utf-8") as f:
                json.dump(all_items, f, indent=2, ensure_ascii=False)
            with open(os.path.join(od, "_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"time": datetime.datetime.now().isoformat(), "source": anno_path,
                    "total": total, "by_label": stats}, f, indent=2, ensure_ascii=False)
        print(f"[导出] {total} 条 → {od}")
        self._jr({"output_dir": od, "total": total, "stats": stats})

    def _jr(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="标注审核+修改工具")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"{'='*50}\n  标注审核+修改工具\n{'='*50}")
    print(f"  http://localhost:{args.port}")
    print(f"  http://<IP>:{args.port}")
    print(f"{'='*50}\n  Ctrl+C 停止\n")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\n停止"); server.server_close()

if __name__ == "__main__": main()
