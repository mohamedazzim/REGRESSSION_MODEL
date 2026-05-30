document.addEventListener('DOMContentLoaded', ()=>{
  const form = document.getElementById('predict-form');
  const result = document.getElementById('result');

  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    result.textContent = 'Requesting prediction...';
    const fd = new FormData(form);
    const payload = {};
    for(const [k,v] of fd.entries()) payload[k]=Number(v);

    try{
      const res = await fetch('/api/predict',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({data: payload})
      });
      if(!res.ok) throw new Error(await res.text());
      const json = await res.json();
      result.innerHTML = `<strong>Predicted price:</strong> <span class="price">${Number(json.prediction).toFixed(2)}</span>`;
    }catch(err){
      console.error(err);
      result.textContent = 'Error: ' + (err.message||err);
    }
  });
});
