# HeterSNE
## Usage Example
### HeterSNE
CiteSeer 
```javascript 
python main.py --dataset CiteSeer --epochs 700 --lr 0.001 --lr_gamma 0.0005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0 --task node_classification --str_aug ANA --layer 1 --L 1 --alpha 0.9
```
Wisconsin
```javascript 
python main.py --dataset wisconsin --epochs 500 --lr 0.005 --lr_gamma 0.005 --weight_decay 0.0005 --hidden_size 512 --output_size 512 --dropout 0.3 --task node_classification --str_aug ANA --layer 4 5 --L 2 --alpha 0.9
```
## Node Classification Results
model	|Cora	|CiteSeer	|PubMed|Cornell|Texas	|Wisconsin	|Actor
------ | -----  |----------- |---|--- | -----  |----------- |-------
HeterSNE|	83.0% |	72.7%|	86.2%|73.9%|	74.7% |	74.4%|36.8%

