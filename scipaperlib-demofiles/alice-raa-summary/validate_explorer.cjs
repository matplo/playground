// DOM-level logic checks without starting or controlling a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const page=fs.readFileSync('alice_raa_explorer.html','utf8');
const dataText=page.match(/<script id="dataset" type="application\/json">([\s\S]*?)<\/script>/)[1];
const script=page.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{
 constructor(tag,id=''){this.tag=tag;this.id=id;this.children=[];this.attributes={};this.value='';this.checked=false;this.textContent='';this.innerHTML='';this.dataset={};this.style={};this.handlers={};this.offsetHeight=180}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=[...nodes];this.textContent='';this.innerHTML=''}
 setAttribute(k,v){this.attributes[k]=String(v)}
 addEventListener(k,fn){this.handlers[k]=fn}
 click(){if(this.onclick)this.onclick()}
}
const ids={};for(const m of page.matchAll(/\bid="([^"]+)"/g))ids[m[1]]=new Element('div',m[1]);
ids.dataset.textContent=dataText;ids.logx.checked=true;ids.syst.checked=true;
function encode(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;')}
function serialize(n){return `<${n.tag}${Object.entries(n.attributes).map(([k,v])=>` ${k}="${encode(v)}"`).join('')}>${encode(n.textContent)}${n.children.map(serialize).join('')}</${n.tag}>`}
const context={console,document:{getElementById:id=>ids[id],createElement:t=>new Element(t),createElementNS:(ns,t)=>new Element(t)},window:{innerWidth:1440,innerHeight:1000},Blob,URL,setTimeout,XMLSerializer:class{serializeToString(n){return serialize(n)}}};
vm.createContext(context);vm.runInContext(script,context);
const run=s=>vm.runInContext(s,context);let cases=0;
function checkSVG(){const s=run('new XMLSerializer().serializeToString(currentSVG)');assert(!/NaN|Infinity|undefined/.test(s));assert(s.includes('<circle'));cases++;return s}
checkSVG();assert(run('selected.size')===8);
for(const name of Object.keys(JSON.parse(dataText).presets)){
 ids.preset.value=name;ids.preset.onchange();checkSVG();assert(run('selected.size')>0);
}
ids.logx.checked=false;ids.logx.handlers.change();checkSVG();
ids.syst.checked=false;ids.syst.handlers.change();checkSVG();
ids.xmin.value='1';ids.xmax.value='20';ids.xmax.handlers.change();checkSVG();
ids.xmin.value='30';ids.xmax.value='20';ids.xmax.handlers.change();assert(ids.plots.textContent.includes('valid pT range'));cases++;
ids.resetRange.onclick();checkSVG();
ids.clear.onclick();assert(run('selected.size')===0);assert(ids.plots.innerHTML.includes('Select measurements'));cases++;
ids.system.value='p-Pb';ids.family.value='Jets';ids.search.value='';ids.system.handlers.input();assert(run('visible.every(c=>c.system==="p-Pb"&&c.family==="Jets")'));ids.selectVisible.onclick();assert(run('selected.size')===6);checkSVG();
ids.system.value='';ids.family.value='';ids.search.value='pb_d0';ids.search.handlers.input();assert(run('visible.length')===1);cases++;
// Checkbox interaction adds one measured RAA and hence another observable panel.
const checkbox=ids.choices.children[0].children[0];checkbox.checked=true;checkbox.onchange();assert(run('selected.has("pb_d0")'));checkSVG();
run('globalThis.downloads=[]; download=(name,content,type)=>downloads.push({name,content,type})');
ids.jsonDownload.onclick();ids.csvDownload.onclick();ids.svgDownload.onclick();
const downloads=run('downloads');assert(downloads.length===3);const out=JSON.parse(downloads[0].content);assert(out.curves.length===7);assert(out.curves.every(c=>c.sha256&&c.points[0].original_y));assert(downloads[1].content.split('\n').length===1+out.curves.reduce((n,c)=>n+c.points.length,0));cases+=3;
// All 65 curves across distinct observables exercise every dataset.
run('selected=new Set(DATA.curves.map(c=>c.key));render()');checkSVG();assert(ids.sources.children.length===65);cases++;
ids.preset.value='Central Pb-Pb · jets and photons';ids.preset.onchange();ids.logx.checked=true;ids.logx.handlers.change();
fs.mkdirSync('qa',{recursive:true});fs.writeFileSync('qa/explorer_generated.svg',checkSVG());
fs.writeFileSync('qa/explorer_validation.json',JSON.stringify({passed:true,cases,method:'Node vm with minimal DOM; actual explorer script executed',coverage:['all presets','all 65 datasets','linear/log axes','uncertainty toggle','range validation','filters','select/clear','checkbox selection','SVG/CSV/JSON exports','source cards'],limitation:'No browser engine or native browser UI was available; CSS rendering and real download prompts were not exercised.'},null,2));
console.log(`${cases} explorer logic checks passed.`);
