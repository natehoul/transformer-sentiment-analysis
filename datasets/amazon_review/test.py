from amazonReviewsDatasets import AmazonReviewsDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

my_dataset = AmazonReviewsDataset('tahi')

print(f"Dataset made: {len(my_dataset)}")

# NOTE: Formation of the DataLoader works fine, but iteration over it breaks completely
# May or may not be related to the bug in bertEmbed idk
my_loader = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=1)

print('Dataloader made')

i = 0
iters = 1000
for data, label in tqdm(my_dataset, total=iters):
    i += 1
    if i > iters:
        break

