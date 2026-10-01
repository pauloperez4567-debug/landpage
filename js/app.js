const V=window.VET,$=(s,r=document)=>r.querySelector(s);
const pages=[["index.html","Inicio"],["servicios.html","Servicios"],["alimentacion.html","Alimentación"],["nosotros.html","Nosotros"],["contacto.html","Contacto"]];
const cur=location.pathname.split("/").pop()||"index.html";
$("header").innerHTML=`<div class="wrap"><a class="logo" href="index.html">🐾 ${V.nombre}</a><nav>${pages.map(([h,t])=>`<a href="${h}" class="${h===cur?"on":""}">${t}</a>`).join("")}</nav></div>`;
$("footer").innerHTML=`<div class="wrap">📞 Recepción: <a href="tel:${V.telefono.replace(/[^\d+]/g,"")}">${V.telefono}</a> · 📍 ${V.direccion} · 🕘 ${V.horario}</div>`;
document.querySelectorAll("[data-vet]").forEach(e=>e.textContent=V[e.dataset.vet]);
document.querySelectorAll("[data-tel]").forEach(e=>e.href="tel:"+V.telefono.replace(/[^\d+]/g,""));
const mapEl=$("#mapa");if(mapEl)mapEl.innerHTML=V.mapaEmbed?`<iframe class="map" src="${V.mapaEmbed}" loading="lazy" title="Mapa"></iframe>`:`<div class="ph">Mapa pendiente.<br>Pega la URL de Google Maps en js/config.js (mapaEmbed).</div>`;

const COLORS=["#2f855a","#f6ad55","#4299e1","#ed64a6","#9f7aea"];
function bars(el,data,unit=""){const w=520,h=40*data.length+10,max=Math.max(...data.map(d=>d[1]));
 el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img">${data.map((d,i)=>{const bw=(w-190)*d[1]/max;return `<text class="bar-lbl" x="0" y="${i*40+24}">${d[0]}</text><rect x="110" y="${i*40+8}" width="${bw}" height="26" rx="6" fill="${COLORS[i%5]}"/><text class="bar-val" x="${118+bw}" y="${i*40+26}">${d[1]}${unit}</text>`}).join("")}</svg>`}
function donut(el,data){const tot=data.reduce((a,d)=>a+d[1],0),r=70,c=2*Math.PI*r;let off=0;
 el.innerHTML=`<svg viewBox="0 0 200 200" role="img" style="max-width:260px;display:block;margin:auto"><g transform="rotate(-90 100 100)">${data.map((d,i)=>{const l=c*d[1]/tot,s=`<circle cx="100" cy="100" r="${r}" fill="none" stroke="${COLORS[i%5]}" stroke-width="32" stroke-dasharray="${l} ${c-l}" stroke-dashoffset="${-off}"/>`;off+=l;return s}).join("")}</g><text x="100" y="106" text-anchor="middle" class="bar-val" style="font-size:20px">${tot}%</text></svg><div class="legend">${data.map((d,i)=>`<span><i style="background:${COLORS[i%5]}"></i>${d[0]} ${d[1]}%</span>`).join("")}</div>`}
function line(el,labels,vals){const w=520,h=220,p=30,max=Math.max(...vals)*1.15,x=i=>p+i*(w-2*p)/(vals.length-1),y=v=>h-p-(h-2*p)*v/max;
 el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img"><polyline fill="none" stroke="${COLORS[0]}" stroke-width="3" points="${vals.map((v,i)=>x(i)+","+y(v)).join(" ")}"/>${vals.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${COLORS[0]}"/><text class="bar-val" x="${x(i)}" y="${y(v)-10}" text-anchor="middle">${v}</text><text class="bar-lbl" x="${x(i)}" y="${h-8}" text-anchor="middle">${labels[i]}</text>`).join("")}</svg>`}
const g=id=>document.getElementById(id);
if(g("c-tipo"))donut(g("c-tipo"),[["Seca",58],["Húmeda",22],["Natural/BARF",12],["Dietética",8]]);
if(g("c-marca"))bars(g("c-marca"),[["Perros adultos",42],["Cachorros",18],["Gatos adultos",25],["Gatitos",8],["Senior",7]],"%");
if(g("c-mes"))line(g("c-mes"),["Ene","Feb","Mar","Abr","May","Jun"],[120,135,128,150,162,175]);
if(g("c-kg"))bars(g("c-kg"),[["Perro grande",14],["Perro mediano",8],["Perro pequeño",3],["Gato",4]]," kg");
