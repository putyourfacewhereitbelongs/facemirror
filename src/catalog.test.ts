import {describe,expect,it} from 'vitest';
import {categories,demos,getDemo} from './catalog';

describe('experiment catalog contracts',()=>{
  it('has unique routable workspaces',()=>{
    const slugs=demos.map(d=>d.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
    for(const demo of demos)expect(getDemo(demo.slug)).toBe(demo);
  });
  it('fully describes every workspace',()=>{
    for(const demo of demos){
      expect(demo.title.length).toBeGreaterThan(2);
      expect(demo.description.length).toBeGreaterThan(20);
      expect(demo.capabilities.length).toBeGreaterThanOrEqual(3);
      expect(categories).toContain(demo.category);
      expect(demo.accent).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
  it('keeps real interactive engines discoverable',()=>{
    expect(demos.some(d=>d.slug==='mediapipe-mesh')).toBe(true);
    expect(demos.some(d=>d.slug==='liveportrait')).toBe(true);
    expect(demos.some(d=>d.slug==='pipeline')).toBe(true);
  });
});
