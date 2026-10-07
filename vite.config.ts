import { resolve } from 'node:path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode})=>{
  const env=loadEnv(mode,process.cwd(),'');
  return {plugins:[react()],
    build:{rollupOptions:{input:{main:resolve(__dirname,'index.html'),puppet:resolve(__dirname,'puppet.html')}}},
    server:{host:true,allowedHosts:true,proxy:{'/ws':{target:env.GPU_BACKEND_URL||'http://127.0.0.1:8000',ws:true},'/health':{target:env.GPU_BACKEND_URL||'http://127.0.0.1:8000'}}}};
});
