import os, subprocess

pdf_file = r"Case 1.pdf"

#os.chdir(pdf_file)

pdftoppm_path = r"execsrc/poppler-0.67.0_x86/bin/pdftoppm.exe"

subprocess.Popen('"%s" -jpeg -f 1 -l 1 "%s" pdfimage' % (pdftoppm_path, pdf_file), shell=True)