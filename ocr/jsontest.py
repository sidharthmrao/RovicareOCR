import subprocess
import json

directory = "/mnt/c/Users/sidha/Desktop/ocrtesting"
file = "completed-cprform.pdf"
parts = file.split("/")

print("Running...")

convert = subprocess.Popen(f"docker run -it --rm -v {directory}:/data convertmed {file}", shell=True, stdout=subprocess.PIPE)
convert.wait()
out, err = convert.communicate()

with open(f"output.json", "w") as file:
    output = json.loads(out)
    json.dump(output, file, indent=4)

print("Complete")
