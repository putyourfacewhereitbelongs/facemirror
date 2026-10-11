import { useCallback, useEffect, useRef, useState } from 'react';
import { FaceMesh, type Results } from '@mediapipe/face_mesh';

export type Point = {x:number;y:number;z:number};
export type Signals = { jawOpen:number; smile:number; blinkLeft:number; blinkRight:number; brow:number; yaw:number; roll:number };
export type EngineState = {
  status:'idle'|'loading'|'ready'|'error'; fps:number; landmarks:Point[]|null; signals:Signals;
  error?:string; running:boolean;
};
const zero:Signals={jawOpen:0,smile:0,blinkLeft:0,blinkRight:0,brow:0,yaw:0,roll:0};
const dist=(a:Point,b:Point)=>Math.hypot(a.x-b.x,a.y-b.y);
const clamp=(n:number)=>Math.max(0,Math.min(1,n));
function derive(p:Point[]):Signals {
  const face=dist(p[234],p[454])||1;
  const eyeL=dist(p[159],p[145])/(dist(p[33],p[133])||1);
  const eyeR=dist(p[386],p[374])/(dist(p[362],p[263])||1);
  const mouthW=dist(p[61],p[291])/face;
  return {
    jawOpen:clamp((dist(p[13],p[14])/face-.005)*16), smile:clamp((mouthW-.32)*8),
    blinkLeft:clamp(1-eyeL*4.8), blinkRight:clamp(1-eyeR*4.8),
    brow:clamp((dist(p[105],p[159])/face-.10)*8),
    yaw:Math.max(-1,Math.min(1,((p[1].x-(p[234].x+p[454].x)/2)/face)*3)),
    roll:Math.max(-1,Math.min(1,(p[263].y-p[33].y)*4))
  };
}

export function useFaceEngine(){
  const videoRef=useRef<HTMLVideoElement>(null); const mesh=useRef<FaceMesh|null>(null);
  const stream=useRef<MediaStream|null>(null); const raf=useRef(0); const busy=useRef(false);
  const frames=useRef({time:performance.now(),count:0});
  const [state,setState]=useState<EngineState>({status:'idle',fps:0,landmarks:null,signals:zero,running:false});
  const ensureMesh=useCallback(()=>{
    if(mesh.current)return mesh.current;
    const m=new FaceMesh({locateFile:(f)=>`/face_mesh/${f}`});
    m.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.55,minTrackingConfidence:.5});
    m.onResults((r:Results)=>{
      const points=(r.multiFaceLandmarks?.[0]||[]) as Point[];
      frames.current.count++;
      const elapsed=performance.now()-frames.current.time; let fps:number|undefined;
      if(elapsed>500){fps=Math.round(frames.current.count*1000/elapsed);frames.current={time:performance.now(),count:0};}
      setState(s=>({...s,landmarks:points.length?points:null,signals:points.length?derive(points):zero,fps:fps??s.fps}));
    }); mesh.current=m; return m;
  },[]);
  const loop=useCallback(async()=>{
    const v=videoRef.current;
    if(v&&mesh.current&&v.readyState>=2&&!busy.current){ busy.current=true; try{await mesh.current.send({image:v});}finally{busy.current=false;} }
    raf.current=requestAnimationFrame(loop);
  },[]);
  const detectStill=useCallback(async(image:HTMLImageElement)=>{
    const detector=new FaceMesh({locateFile:(f)=>`/face_mesh/${f}`});
    detector.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.45,minTrackingConfidence:.45});
    return new Promise<Point[]|null>((resolve)=>{
      let settled=false;
      const finish=(points:Point[]|null)=>{if(settled)return;settled=true;clearTimeout(timeout);detector.close().catch(()=>{});resolve(points)};
      const timeout=setTimeout(()=>finish(null),8000);
      detector.onResults((r:Results)=>{const points=(r.multiFaceLandmarks?.[0]||[]) as Point[];finish(points.length?points:null)});
      detector.send({image}).catch(()=>finish(null));
    });
  },[]);
  const start=useCallback(async()=>{
    try{
      setState(s=>({...s,status:'loading',error:undefined}));
      ensureMesh();
      stream.current=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:1280},height:{ideal:720},frameRate:{ideal:60}},audio:false});
      if(videoRef.current){videoRef.current.srcObject=stream.current;await videoRef.current.play();}
      setState(s=>({...s,status:'ready',running:true})); cancelAnimationFrame(raf.current); raf.current=requestAnimationFrame(loop);
    }catch(e){setState(s=>({...s,status:'error',running:false,error:e instanceof Error?e.message:'Camera could not start'}));}
  },[loop,ensureMesh]);
  const stop=useCallback(()=>{stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;cancelAnimationFrame(raf.current);setState(s=>({...s,running:false,status:'idle',fps:0}));},[]);
  useEffect(()=>()=>{stream.current?.getTracks().forEach(t=>t.stop());cancelAnimationFrame(raf.current);mesh.current?.close();},[]);
  return {videoRef,state,start,stop,detectStill};
}
