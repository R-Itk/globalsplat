function initExportPdf() {
  const btn = document.getElementById('export-pdf-btn');
  if (!btn || typeof html2pdf === 'undefined') return;

  btn.addEventListener('click', () => {
    // const element = document.querySelector('main') || document.body;
    const element = document.body;
    const opt = {
      margin: 0.5,
      filename: 'globalsplat-page.pdf',
      image: { type: 'jpeg', quality: 0.95 },
      html2canvas: { scale: 2, useCORS: true },
      jsPDF: { unit: 'in', format: 'a4', orientation: 'portrait' },
    };

    html2pdf().from(element).set(opt).save();
  });
}

