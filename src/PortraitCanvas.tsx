import { useEffect, useMemo, useRef, useState } from 'react';
import { FACEMESH_CONTOURS, FACEMESH_TESSELATION } from '@mediapipe/face_mesh';
import type { Demo } from './catalog';
import type { EngineState, Point } from './useFaceEngine';

type Expression='live'|'smile'|'frown'|'o';
type Props={src:string;state:EngineState;prefs:{sensitivity:number;pose:number;mouth:number;brightness:number;contrast:number;saturation:number;identity:boolean;reduceMotion:boolean;mesh:boolean};demo:Demo;expression:Expression;onExpression:(e:Expression)=>void;detectStill:(img:HTMLImageElement)=>Promise<Point[]|null>;onCanvas:(c:HTMLCanvasElement|null)=>void;neuralFrame?:ImageBitmap|null;neuralActive?:boolean};
type XY={x:number;y:number};
const cycles:Expression[]=['smile','frown','o','live'];
function affine(ctx:CanvasRenderingContext2D,img:HTMLImageElement,s:XY[],d:XY[]){
 const [p,q,r]=s,[u,v,w]=d;const den=p.x*(q.y-r.y)+q.x*(r.y-p.y)+r.x*(p.y-q.y);if(Math.abs(den)<.01)return;
 const a=(u.x*(q.y-r.y)+v.x*(r.y-p.y)+w.x*(p.y-q.y))/den;
 const c=(u.x*(r.x-q.x)+v.x*(p.x-r.x)+w.x*(q.x-p.x))/den;
 const e=(u.x*(q.x*r.y-r.x*q.y)+v.x*(r.x*p.y-p.x*r.y)+w.x*(p.x*q.y-q.x*p.y))/den;
 const b=(u.y*(q.y-r.y)+v.y*(r.y-p.y)+w.y*(p.y-q.y))/den;
 const dd=(u.y*(r.x-q.x)+v.y*(p.x-r.x)+w.y*(q.x-p.x))/den;
 const f=(u.y*(q.x*r.y-r.x*q.y)+v.y*(r.x*p.y-p.x*r.y)+w.y*(p.x*q.y-q.x*p.y))/den;
 ctx.save();ctx.beginPath();ctx.moveTo(u.x,u.y);ctx.lineTo(v.x,v.y);ctx.lineTo(w.x,w.y);ctx.closePath();ctx.clip();ctx.setTransform(a,b,c,dd,e,f);ctx.drawImage(img,0,0);ctx.restore();
}
function oral(ctx:CanvasRenderingContext2D,pts:XY[],amount:number,mode:Expression){
 if(amount<.08||!pts[13]||!pts[14])return;const l=pts[61],r=pts[291],top=pts[13],bot=pts[14];const cx=(l.x+r.x)/2,cy=(top.y+bot.y)/2;const w=Math.max(12,Math.hypot(r.x-l.x)*.72),h=Math.max(5,w*(.13+amount*.34));
 ctx.save();ctx.beginPath();ctx.ellipse(cx,cy,w/2,h/2,0,0,Math.PI*2);ctx.clip();let g=ctx.createRadialGradient(cx,cy-h*.1,1,cx,cy,w*.55);g.addColorStop(0,'#741f2c');g.addColorStop(.55,'#321016');g.addColorStop(1,'#100407');ctx.fillStyle=g;ctx.fillRect(cx-w/2,cy-h/2,w,h);
 const teethH=h*Math.min(.46,.22+amount*.12);let tg=ctx.createLinearGradient(0,cy-h/2,0,cy-h/2+teethH);tg.addColorStop(0,'#fffdf1');tg.addColorStop(.6,'#eee9d9');tg.addColorStop(1,'#b9aa99');ctx.fillStyle=tg;ctx.beginPath();ctx.roundRect(cx-w*.43,cy-h*.47,w*.86,teethH,[2,2,5,5]);ctx.fill();ctx.strokeStyle='rgba(90,72,65,.22)';ctx.lineWidth=.65;for(let i=-3;i<=3;i++){ctx.beginPath();ctx.moveTo(cx+i*w*.105,cy-h*.45);ctx.lineTo(cx+i*w*.095,cy-h*.45+teethH*.83);ctx.stroke()}
 if(amount>.32||mode==='o'){let tongue=ctx.createRadialGradient(cx,cy+h*.36,1,cx,cy+h*.4,w*.38);tongue.addColorStop(0,'#d96c79');tongue.addColorStop(.72,'#9d3f51');tongue.addColorStop(1,'#65232e');ctx.fillStyle=tongue;ctx.beginPath();ctx.ellipse(cx,cy+h*.42,w*.34,h*.23,0,Math.PI,Math.PI*2);ctx.fill();ctx.strokeStyle='rgba(83,25,38,.35)';ctx.beginPath();ctx.moveTo(cx,cy+h*.25);ctx.quadraticCurveTo(cx-w*.02,cy+h*.38,cx,cy+h*.49);ctx.stroke()}
 ctx.fillStyle='rgba(255,255,255,.32)';ctx.beginPath();ctx.ellipse(cx-w*.17,cy-h*.29,w*.12,Math.max(1,h*.035),-.08,0,Math.PI*2);ctx.fill();ctx.restore();
}
function drawMappedMesh(ctx:CanvasRenderingContext2D,points:Point[],map:(p:Point)=>XY,accent:string){
 ctx.save();ctx.strokeStyle=accent;ctx.globalAlpha=.16;ctx.lineWidth=.65;ctx.beginPath();for(let i=0;i<FACEMESH_TESSELATION.length;i+=3){const [a,b]=FACEMESH_TESSELATION[i];if(!points[a]||!points[b])continue;const p=map(points[a]),q=map(points[b]);ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y)}ctx.stroke();ctx.globalAlpha=.72;ctx.lineWidth=1.6;ctx.beginPath();for(const [a,b] of FACEMESH_CONTOURS){if(!points[a]||!points[b])continue;const p=map(points[a]),q=map(points[b]);ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y)}ctx.stroke();ctx.globalAlpha=.95;ctx.fillStyle='#72f2ff';for(let i=0;i<Math.min(478,points.length);i++){const p=map(points[i]);ctx.beginPath();ctx.arc(p.x,p.y,i%10===0?2.35:1.25,0,Math.PI*2);ctx.fill()}ctx.restore();
}
function drawTrackingMesh(ctx:CanvasRenderingContext2D,points:Point[],w:number,h:number,accent:string){drawMappedMesh(ctx,points,p=>({x:(1-p.x)*w,y:p.y*h}),accent)}
function drawSourceMesh(ctx:CanvasRenderingContext2D,points:Point[],ox:number,oy:number,w:number,h:number,accent:string){drawMappedMesh(ctx,points,p=>({x:ox+p.x*w,y:oy+p.y*h}),accent)}
export default function PortraitCanvas({src,state,prefs,demo,expression,onExpression,detectStill,onCanvas,neuralFrame,neuralActive}:Props){
 const canvas=useRef<HTMLCanvasElement>(null),imgRef=useRef<HTMLImageElement|null>(null),source=useRef<Point[]|null>(null),baseline=useRef<Point[]|null>(null);const [ready,setReady]=useState(false);
 const triangles=useMemo(()=>{const e=FACEMESH_TESSELATION as [number,number][];const out:number[][]=[];for(let i=0;i+2<e.length;i+=3)out.push([e[i][0],e[i][1],e[i+1][1]]);return out},[]);
 useEffect(()=>{baseline.current=null;source.current=null;setReady(false);const img=new Image();img.crossOrigin='anonymous';img.onload=async()=>{imgRef.current=img;source.current=await detectStill(img);setReady(true)};img.src=src},[src,detectStill]);
 useEffect(()=>{onCanvas(canvas.current);return()=>onCanvas(null)},[onCanvas]);
 useEffect(()=>{if(!state.running)baseline.current=null;else if(state.landmarks&&!baseline.current)baseline.current=state.landmarks.map(p=>({...p}))},[state.landmarks,state.running]);
 useEffect(()=>{let frame=0;const render=()=>{const c=canvas.current,img=imgRef.current;if(!c||!img){frame=requestAnimationFrame(render);return}const ctx=c.getContext('2d')!,W=c.width,H=c.height;ctx.clearRect(0,0,W,H);if(neuralActive&&neuralFrame){ctx.fillStyle='#090b12';ctx.fillRect(0,0,W,H);ctx.drawImage(neuralFrame,0,0,W,H);if(prefs.mesh&&state.landmarks)drawTrackingMesh(ctx,state.landmarks,W,H,demo.accent);frame=requestAnimationFrame(render);return}const scale=Math.min(W/img.width,H/img.height),dw=img.width*scale,dh=img.height*scale,ox=(W-dw)/2,oy=(H-dh)/2;ctx.fillStyle='#090b12';ctx.fillRect(0,0,W,H);ctx.filter=`brightness(${prefs.brightness}%) contrast(${prefs.contrast}%) saturate(${prefs.saturation}%)`;ctx.drawImage(img,ox,oy,dw,dh);
 const sp=source.current,live=state.landmarks,base=baseline.current;if(sp){const srcPts=sp.map(p=>({x:p.x*img.width,y:p.y*img.height}));const dst=sp.map(p=>({x:ox+p.x*dw,y:oy+p.y*dh}));if(live&&base&&!prefs.reduceMotion){const gain=prefs.sensitivity/100;const faceW=Math.abs(base[454].x-base[234].x)||1;const sourceW=Math.abs(sp[454].x-sp[234].x)*dw;for(let i=0;i<Math.min(468,sp.length,live.length);i++){const dx=(live[i].x-base[i].x)/faceW*sourceW*gain;const dy=(live[i].y-base[i].y)/faceW*sourceW*gain;dst[i].x+=dx;dst[i].y+=dy}}
 const mouthW=Math.abs(dst[291].x-dst[61].x),manual=expression!=='live';if(manual){const ids=[61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78,191,80,81,82,13,312,311,310,415];if(expression==='smile'){dst[61].x-=mouthW*.12;dst[61].y-=mouthW*.07;dst[291].x+=mouthW*.12;dst[291].y-=mouthW*.07}else if(expression==='frown'){dst[61].y+=mouthW*.1;dst[291].y+=mouthW*.1;dst[13].y-=mouthW*.025;dst[14].y+=mouthW*.025}else if(expression==='o'){ids.forEach(i=>{dst[i].x+=(dst[1].x-dst[i].x)*.28});dst[13].y-=mouthW*.12;dst[14].y+=mouthW*.18}}
 if(prefs.identity)triangles.forEach(t=>{if(t.every(i=>srcPts[i]&&dst[i]))affine(ctx,img,t.map(i=>srcPts[i]),t.map(i=>dst[i]))});const jaw=manual?(expression==='o'?.82:expression==='frown'?.18:.12):state.signals.jawOpen*prefs.mouth/100;oral(ctx,dst,jaw,expression);
 if(prefs.identity&&state.running){ctx.strokeStyle='rgba(255,255,255,.08)';ctx.lineWidth=1;ctx.beginPath();ctx.ellipse(dst[1].x,dst[1].y,mouthW*1.28,mouthW*1.7,0,0,Math.PI*2);ctx.stroke()}}
 ctx.filter='none';if(prefs.mesh){if(state.landmarks)drawTrackingMesh(ctx,state.landmarks,W,H,demo.accent);else if(source.current)drawSourceMesh(ctx,source.current,ox,oy,dw,dh,demo.accent)}ctx.fillStyle=demo.accent;ctx.fillRect(0,H-3,W*(state.running?Math.max(.03,state.fps/60):.03),3);frame=requestAnimationFrame(render)};frame=requestAnimationFrame(render);return()=>cancelAnimationFrame(frame)},[state,prefs,expression,demo.accent,triangles,neuralFrame,neuralActive]);
 return <canvas ref={canvas} width={900} height={900} className="render-canvas" onClick={()=>onExpression(cycles[(cycles.indexOf(expression)+1)%cycles.length])} title="Click the face to cycle smile, frown, O, and live expressions" data-ready={ready}/>;
}
