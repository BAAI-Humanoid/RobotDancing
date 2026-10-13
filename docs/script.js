const backgroundVideo=document.querySelector('.teaser video');
if(matchMedia('(prefers-reduced-motion: reduce)').matches)backgroundVideo.pause();
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
