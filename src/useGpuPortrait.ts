import { useCallback, useEffect, useRef, useState } from 'react';
import type { NeuralState } from './useLivePortrait';

export function useGpuPortrait(videoRef:React.RefObject<HTMLVideoElement>){
 const ws=useRef<WebSocket|null>(null),active=useRef(false),capture=useRef<HTMLCanvasElement|null>(null),frameRef=useRef<ImageBitmap|null>(null),resolveStart=useRef<(()=>void)|null>(null),rejectStart=useRef<((e:Error)=>void)|null>(null);
 const [state,setState]=useState<NeuralState>({status:'idle',progress:0,detail:'Looking for TensorRT GPU service',inferenceMs:0,frame:null});
 const sendFrame=useCallback(()=>{const socket=ws.current,v=videoRef.current;if(!active.current||!socket||socket.readyState!==WebSocket.OPEN||!v||v.readyState<2)return;const c=capture.current||(capture.current=document.createElement('canvas'));c.width=512;c.height=512;const x=c.getContext('2d')!,size=Math.min(v.videoWidth,v.videoHeight),sx=(v.videoWidth-size)/2,sy=Math.max(0,(v.videoHeight-size)/2-size*.08);x.save();x.translate(512,0);x.scale(-1,1);x.drawImage(v,sx,sy,size,size,0,0,512,512);x.restore();c.toBlob(blob=>{if(blob&&active.current&&socket.readyState===WebSocket.OPEN)socket.send(blob)},'image/jpeg',.88)},[videoRef]);
 const stop=useCallback(()=>{active.current=false;ws.current?.close();ws.current=null;setState(s=>({...s,status:s.frame?'ready':'idle',detail:'GPU stream stopped'}))},[]);
 const start=useCallback(async(source:string)=>{
   stop();active.current=true;setState(s=>({...s,status:'loading',progress:5,detail:'Connecting to TensorRT GPU renderer'}));
   const protocol=location.protocol==='https:'?'wss:':'ws:',socket=new WebSocket(`${protocol}//${location.host}/ws/liveportrait`);socket.binaryType='blob';ws.current=socket;
   const ready=new Promise<void>((resolve,reject)=>{resolveStart.current=resolve;rejectStart.current=reject;const timeout=setTimeout(()=>reject(new Error('GPU renderer connection timed out')),5000);socket.onopen=async()=>{clearTimeout(timeout);try{const blob=await fetch(source).then(r=>r.blob());socket.send(blob);setState(s=>({...s,progress:35,detail:'Encoding source identity on GPU'}))}catch(e){reject(e instanceof Error?e:new Error(String(e)))}};socket.onerror=()=>{clearTimeout(timeout);reject(new Error('GPU renderer is unavailable'))}});
   socket.onmessage=async event=>{if(typeof event.data==='string'){const message=JSON.parse(event.data);if(message.type==='source_ready'){setState(s=>({...s,status:'running',progress:100,detail:'FasterLivePortrait · TensorRT GPU'}));resolveStart.current?.();resolveStart.current=null;sendFrame()}else if(message.type==='timing')setState(s=>({...s,inferenceMs:message.ms,detail:`TensorRT neural render · ${message.ms} ms`}));else if(message.type==='error'){const error=new Error(message.message);rejectStart.current?.(error);rejectStart.current=null;setState(s=>({...s,status:'error',detail:message.message}))}return}const bitmap=await createImageBitmap(event.data as Blob),old=frameRef.current;frameRef.current=bitmap;setState(s=>({...s,status:'running',frame:bitmap,progress:100}));setTimeout(()=>old?.close(),100);requestAnimationFrame(sendFrame)};
   socket.onclose=()=>{if(active.current)setState(s=>({...s,status:'error',detail:'GPU renderer disconnected'}))};
   try{await ready}catch(e){active.current=false;socket.close();throw e}
 },[sendFrame,stop]);
 useEffect(()=>()=>{active.current=false;ws.current?.close();frameRef.current?.close()},[]);
 return {state,start,stop};
}
