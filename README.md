# HeterGCL
## Usage Example
### HeterSNE
CiteSeer 
```javascript 
python main.py --dataset CiteSeer --epochs 700 --lr 0.001 --lr_gamma 0.0005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0 --task node_classification --str_aug ANA --layer 1 --L 1 --alpha 0.9
```
Cornell
```javascript 
python main.py --dataset cornell --epochs 500 --lr 0.005 --lr_gamma 0.0005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.3 --task node_classification --str_aug ANA --layer 4 5 --L 2 --alpha 0.85
```
## Node Classification Results
model	|Cornell	|Texas	|Wisconsin|Actor|Cora	|CiteSeer	|Pubmed
------ | -----  |----------- |---|--- | -----  |----------- |-------
HeterGCL|	75.5% |	74.7%|	75.6%|37.2%|	83.0% |	73.0%|86.2%

