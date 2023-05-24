import csv

def write_to_csv(out_dict,outfile='norm_factor.csv'):
    
    with open(outfile,'a+') as f:
        writer=csv.DictWriter(f,out_dict.keys())
        writer.writeheader()
        writer.writerows([out_dict])   