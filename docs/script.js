const backgroundVideo=document.querySelector('.title-hero video');
if(matchMedia('(prefers-reduced-motion: reduce)').matches)backgroundVideo.pause();
const pauseButton=document.getElementById('pause');
function updatePlaybackControl(){const label=backgroundVideo.paused?'Play background video':'Pause background video';pauseButton.textContent=backgroundVideo.paused?'▶':'Ⅱ';pauseButton.setAttribute('aria-label',label);pauseButton.title=label}
backgroundVideo.addEventListener('play',updatePlaybackControl);
backgroundVideo.addEventListener('pause',updatePlaybackControl);
updatePlaybackControl();
pauseButton.addEventListener('click',()=>{if(backgroundVideo.paused)backgroundVideo.play().catch(updatePlaybackControl);else backgroundVideo.pause()});
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
