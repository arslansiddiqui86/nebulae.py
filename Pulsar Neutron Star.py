import moderngl, time
import numpy as np
import pygame as pg

pg.mixer.pre_init(44100, -16, 2, 512); pg.init()
rate, dur, rng = 44100, 14.0, np.random.default_rng(7)
t = np.linspace(0, dur, int(rate * dur), dtype=np.float32)
env = np.minimum(1.0, t / 1.0) * np.minimum(1.0, (dur - t) / 1.0)
swell = np.clip((t - 0.8) / 3.2, 0.0, 1.0) ** 1.8
ph_p = (t * 11.2) % 1.0
tick = np.exp(-ph_p * 24.0) * swell
noise = rng.uniform(-1.0, 1.0, len(t)).astype(np.float32)
thump = np.sin(2.0 * np.pi * (140.0 * ph_p + 18.0 * (1.0 - np.exp(-ph_p * 30.0))))
sig = tick * (0.72 * noise + 0.55 * thump) + 0.03 * np.convolve(noise, np.ones(8) / 8, "same")
pcm = np.clip(31500 * env * sig, -32767, 32767).astype(np.int16)
snd = pg.sndarray.make_sound(np.column_stack([pcm, pcm]))

pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
W, H = pg.display.Info().current_w, pg.display.Info().current_h
pg.display.set_mode((W, H), pg.OPENGL | pg.DOUBLEBUF | pg.NOFRAME); pg.mouse.set_visible(False)
ctx = moderngl.create_context()

prog = ctx.program(
    vertex_shader="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}",
    fragment_shader="""#version 330
    uniform vec2 u_res; uniform float u_t; out vec4 f_col;
    float h(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
    float ns(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);}
    float fbm(vec2 p){float v=0.,a=.5;mat2 m=mat2(.8,.6,-.6,.8);for(int i=0;i<6;i++){v+=a*ns(p);p=m*p*2.02;a*=.5;}return v;}
    void main(){
        vec2 uv0=(gl_FragCoord.xy-.5*u_res)/u_res.y;
        float zm=1.12-.18*smoothstep(0.,14.,u_t); vec2 uv=uv0*zm;
        float d1=fbm(uv*2.1+vec2(1.4,.2)), d2=fbm(uv*3.8+d1*1.3-vec2(.5,.7));
        vec3 dust=vec3(.24,.17,.10)*smoothstep(.30,.82,d2)*smoothstep(.25,-.30,uv.x);
        vec3 neb=vec3(.03,.13,.38)*smoothstep(.24,.80,fbm(uv*2.4-vec2(.7,-.3)+d1*.8))*smoothstep(-.30,.35,uv.x);
        vec3 glow=vec3(.05,.24,.68)*exp(-dot(uv-vec2(.12,-.02),uv-vec2(.12,-.02))*6.0)*(.6+.4*d2);
        vec3 col=vec3(.008,.014,.032)+dust+neb+glow;
        vec2 sc=gl_FragCoord.xy*zm/2.2; float s=h(floor(sc));
        if(s>.9968){float d=length(fract(sc)-.5),sz=mix(.45,.16,fract(s*31.));col+=mix(vec3(.78,.88,1.),vec3(1.,.90,.74),fract(s*71.))*smoothstep(sz,.02,d)*pow((s-.9968)/.0032,1.1)*(.75+.25*sin(u_t*2.8+s*90.));}
        col*=smoothstep(0.,2.2,u_t);
        float ign=smoothstep(.6,3.8,u_t), pulse=.90+.22*exp(-fract(u_t*11.2)*20.)*ign;
        float ca=.12; vec2 p=mat2(cos(ca),-sin(ca),sin(ca),cos(ca))*uv;
        float ay=abs(p.y), r=length(p), reach=smoothstep(1.2,4.4,u_t)*1.5;
        float shoot=smoothstep(reach+.25,reach-.25,ay)*ign;
        float wave=(p.y>0.)?sin(ay*3.4-u_t*5.4+fbm(vec2(ay*1.8,u_t*.45))*1.4)*pow(ay,1.38)*.085:sin(ay*2.0-u_t*4.2)*pow(ay,1.5)*.02;
        float w=.0022+ay*.068, fil=.68+.32*fbm(vec2((p.x-wave)*22.+u_t*2.0,ay*3.8-u_t*4.8));
        float jet=exp(-pow((p.x-wave)/w,2.))*fil*shoot, spine=exp(-pow((p.x-wave)/(.0013+ay*.015),2.))*shoot;
        float star=exp(-r*68.)*2.4*ign, halo=exp(-r*16.)*.52*ign;
        col+=(vec3(.05,.30,.98)*jet*1.35+vec3(.74,.93,1.)*spine*1.15+vec3(.18,.54,1.)*halo)*pulse+vec3(1.,.99,.96)*star;
        f_col=vec4(pow(max(col/(1.+col*.35),0.),vec3(.90)),1.);}""",
)

quad = ctx.vertex_array(prog, [(ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],dtype=np.float32)), "2f", "p")])
ctx.clear(0,0,0,1); pg.display.flip(); time.sleep(2)
snd.play(); start, clock = time.time(), pg.time.Clock(); prog["u_res"].value = (W, H)
while (el := time.time() - start) < dur:
    if any(e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE for e in pg.event.get()): break
    prog["u_t"].value = el; quad.render(moderngl.TRIANGLE_STRIP); pg.display.flip(); clock.tick(60)
pg.quit()
