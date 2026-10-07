"""PlotPick's library: everything the web application does except the interface.

    pdf_figures         find the figures and tables on a PDF page by their captions
    pdf_backend_pdfium  read a PDF's page facts and render crops with pypdfium2
    figure_images       turn uploads (PDFs, images, ZIP archives) into figure crops
    extraction          the prompt, the model call and the reading of the reply
    exports             R, Excel and LaTeX exports of the extracted rows
    models              the Claude models offered and which API key each uses
    pmc                 PubMed identifiers, the PMID-to-PMCID lookup and PDF download
    process_memory      the process's memory, logged after heavy work

None of these imports Streamlit; streamlit_app.py is the interface over them.
"""
