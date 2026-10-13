const backgroundVideo=document.querySelector('.hero video');
const pauseButton=document.getElementById('pause');
if(matchMedia('(prefers-reduced-motion: reduce)').matches)backgroundVideo.pause();
pauseButton.addEventListener('click',()=>{if(backgroundVideo.paused){backgroundVideo.play().catch(()=>{});pauseButton.textContent='Ⅱ';pauseButton.setAttribute('aria-label','Pause background video')}else{backgroundVideo.pause();pauseButton.textContent='▶';pauseButton.setAttribute('aria-label','Play background video')}});
document.getElementById('copy-citation').addEventListener('click',async()=>{
  const text=document.getElementById('bibtex').textContent;
  const status=document.getElementById('copy-status');
  try{
    if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(text)}
    else{
      const field=document.createElement('textarea');field.value=text;field.style.position='fixed';field.style.opacity='0';document.body.append(field);field.select();
      const copied=document.execCommand('copy');field.remove();if(!copied)throw new Error('Clipboard unavailable');
    }
    status.textContent='BibTeX copied.';
  }catch{status.textContent='Select the BibTeX text to copy it.'}
});
