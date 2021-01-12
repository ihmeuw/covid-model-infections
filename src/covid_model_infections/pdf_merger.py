from typing import List
import os

from PyPDF2 import PdfFileMerger


def pdf_merger(pdfs: List, location_names: List, parent_names: List, outfile: str):
    # how are inputs specified
    assert all([i.endswith('.pdf') for i in pdfs]), 'Not all files passed into `pdfs` are actual PDFs.'
    indir = '/'.join(pdfs[0].split('/')[:-1])

    # compile PDFs
    merger = PdfFileMerger()
    for i, (pdf, location_name, parent_name) in enumerate(zip(pdfs, location_names, parent_names)):
        merger.append(pdf)
        if parent_name in location_names:
            merger.addBookmark(location_name, i+1, parent_name)
        else:
            merger.addBookmark(location_name, i+1)

    # get output file (if already exists, delete before writing new file)
    if outfile is None:
        outfile = f'{indir}/_compiled.pdf'
    else:
        assert outfile.endswith('.pdf'), 'Provided output file is not a PDF.'
    if os.path.exists(outfile):
        os.remove(outfile)

    # write compiled PDF
    merger.write(outfile)
    merger.close()
