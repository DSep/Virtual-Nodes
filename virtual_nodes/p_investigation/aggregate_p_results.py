import csv
import torch

def calculate_average_means():
    # initialize variables to keep track of sums
    ps = [i * 0.05 for i in range(0, 10)]
    ps.extend([i * 0.1 for i in range(6, 15)])
    # initialize a dictionary to keep track of sums and counts for each value of p
    aug_means_p = torch.zeros(len(ps))
    noaug_means_p = torch.zeros(len(ps))
    num_datasets = 9
    # p_sums = {f'{p:.2f}': {'aug_means': 0, 'noaug_means': 0, 'count': 0} for p in ps}
    
    # loop over CSVs with names of the form "average_neighbourhood_consistency_P.csv"
    for i, p in enumerate(ps):
        csv_filename = f"average_neighbourhood_consistency_{p:.2f}.csv"
        
        # open the CSV file
        with open(csv_filename) as csvfile:
            reader = csv.reader(csvfile)
            
            # loop over rows in the CSV
            for row in reader:
                # add the aug_means and noaug_means to the running totals for this value of p
                # print("Row", row)
                # print("Aug", float(row[1]))
                # print("Noaug", float(row[4]))
                aug_means_p[i] += float(row[1])*100
                noaug_means_p[i] += float(row[4])*100
        
        # print(aug_means_p)
        # print(aug_means_p)
        avg_aug_means = aug_means_p[i] / 9
        avg_noaug_means = noaug_means_p[i] / 9
        print(f"For p = {p:.2f}:")
        print(f"  Average aug_means: {avg_aug_means:.5f}")
        print(f"  Average noaug_means: {avg_noaug_means:.5f}")

if __name__ == "__main__":
    calculate_average_means()
