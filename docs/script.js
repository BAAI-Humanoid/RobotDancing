const backgroundVideo=document.querySelector('.hero video');
const pauseButton=document.getElementById('pause');
if(matchMedia('(prefers-reduced-motion: reduce)').matches)backgroundVideo.pause();
pauseButton.addEventListener('click',()=>{if(backgroundVideo.paused){backgroundVideo.play().catch(()=>{});pauseButton.textContent='Ⅱ';pauseButton.setAttribute('aria-label','Pause background video')}else{backgroundVideo.pause();pauseButton.textContent='▶';pauseButton.setAttribute('aria-label','Play background video')}});
