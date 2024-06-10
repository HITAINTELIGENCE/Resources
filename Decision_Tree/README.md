# Decision Tree
## Khái niệm

- Là một thuật toán **non-parametric** supervised learning được sử dụng cho cả classification và regression. Mục tiêu của thuật toán là tìm ra mô hình dự đoán các target bằng cách học các quy tắc quyết định đơn giản từ feature của data.
- Decision Tree có thể xử lý tốt với các feature có dạng category (các feature rời rạc)

![](/Decision_Tree/decision_tree.png)


-  Để có thể xây dựng được một cây quyết định để có thể áp dụng vào trong các bài toán dự đoán 

- Các thành phần chính của cây:
    - non-leaf node: các node chứa câu hỏi
    - root node: non-leaf node đầu tiên
    - leaf-node: node không có node con

- Với mỗi node cần có giá trị target
    - Đối với bài toán classification, target là một class
    - Đối với bài toán regression, target là một giá trị scale

- Giá trị target có thể tính bằng mean các example của node đó.

## Ví dụ

![](/Decision_Tree/iris_decision_tree.png)

![](/Decision_Tree/classification.png)

![](/Decision_Tree/regression.png)

## Learning
- Mấu chốt của decision tree là tìm câu hỏi ở non-leaf node, các câu hỏi này thường là câu hỏi đúng sai, và yếu tố để trả lời các câu hỏi đó là các feature. Vì vậy ý tưởng ở đây là xây dựng các feature chỉ nhận hai giá trị 0 hoặc 1 để đưa ra quyết định:
    - Với category feature, ta có thể sử dụng binary encode với các feature chỉ có 2 giá trị hoặc one-hot với các feature có nhiều giá trị hơn.
    - Với các feature có dạng liện tục thì ta có thể xét ngưỡng (thresh) để quyết định giá trị, và để ngưỡng này sử dụng tham lam sao cho có thể phân loại nhiều nhất.
- Cách xây dựng cây: chọn question ở các node cho đến khi gặp điều kiện dừng

## Evaluation
- Vậy thì làm sao để tìm được câu hỏi cho node. Ta cần một phương pháp hay tiêu chuẩn nào đó để đánh giá được sự phân chia ở một node, phương thức này cần phải thể hiện được mức độ phân tách.
Có hai phương pháp chúng ta có thể đó là:
    - Information gain: được xây dựng từ cross entropy
    - Gini gain: được xây dựng dựa trên Gini Impurity

Bên cạnh đó, chúng cũng áp dụng tư tưởng của Greedy, là thuật toán tham lam, câu hỏi nào hay feature nào được lựa chọn có giá trị gain lớn nhất thì sẽ được chọn.

## Information gain
**Cross Entropy Formula**
    
$$ 
H(\mathbf{S}) = -\sum_{i=1}^m p_i \log(p_i)\quad\quad 
$$

- Trong đó:
    - $p_{i}$: là tỷ lệ các examples trong S thuộc class i

Xét ví dụ với 2 classes:

![](/Decision_Tree/cross_entropy.png)

Từ hình trên chúng ta có thể thấy rừng giá trị của càng lớn hoặc càng nhỏ thì giá trị của $H(S)$ càng nhỏ và ngược lại.


**Information gain formula**

$$
\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right})) 
$$

- $H$: là giá trị cross entropy ở các node phân tách và các node trái, phải.
- $w$: tỷ lệ các example ở nhánh bên trái và bên phải so với số example ở node cha, mục đích là để weight giá trị H ở các node con dựa trên example được phân tách, vì có thể có node con có rất ít example và có node sẽ chứa rất nhiều example.

## Gini gain

- Cách thực hiện giống với **information gain** chỉ khác chúng ta thay **Cross entropy** bằng **Gini**:

$$
Gini(\mathbf{p}) = 1-\sum_{i=1}^m p_i^2
$$

## Loss function

- Có thể thấy với bài toán này không có parameter để có thể update. Hàm loss có mục đích chính là giúp cắt tỉa decision tree (loại bỏ bớt các leaf-node) để tránh overfitting.

- Loss function sẽ được tính dựa trên entropy hoặc gini của các leaf-node.

- Với entropy, chúng ta sẽ có hàm loss như sau:

$$
\mathcal{L} = \sum_{k = 1}^K \frac{|\mathcal{S}_k|}{|\mathcal{S}|}
 H(\mathcal{S}_k)
$$

- Trong đó:
    - $S_{k}$ là số lượng example của leaf-node.
    - $S$ là số lượng example của root-node
    - $H(S_{k})$ là entropy của leaf-node thứ k.

Ta có thể áp dụng *regularization* cho hàm loss:

$$
\mathcal{L} = \sum_{k = 1}^K \frac{|\mathcal{S}_k|}{|\mathcal{S}|}
 H(\mathcal{S}_k) + \lambda K \quad\quad
$$

- Mục đích của việc cắt tỉa là làm giảm giá trị loss function, ta sẽ thử xóa các leaf-node và kiểm tra lại giá trị của hàm loss xem có giảm hay không, kỹ thuật này gọi là Pruning. Xem chi tiết hơn tại machinelearningcoban.com.

## Điều kiện dừng
- Nếu dicision tree đủ sâu thì có thể phân tách hoàn toàn fit với dataset, nhưng điều này có thể dẫn đến overfitting, để tránh điều này ta cần dừng lại đúng lúc, một vài điều kiện sau có thể giúp ích:
    - Đặt thresh cho Entropy hoặc Gini, để ngừng phân tách
    - Đặt giá trị độ sâu của cây giới hạn
    - Tổng leaf-node không được vượt quá một ngưỡng nào đó.

# Random Forest

## Khái niệm
- Là một thuật toán kết hợp với ý tưởng là xây dựng nhiều decision tree, và đưa kết quả cuối cùng dựa trên kết quả của các decision tree, với bài toán classification có thể là class nào có số lượng nhiều nhất trong output của decision tree.

## Mô hình kết hợp (Ensem)

## Cách thức xây dựng thuật toán
- Một điều cần chú ý là với mỗi bộ dataset áp dụng cùng một điều kiện dừng thì chỉ xây được một decision tree, vì vậy ý tưởng ở đây là lấy subset các examples và features, mỗi subset này chúng ta lại xây dựng một cây.

- Chú ý rằng:
    - Khi random examples thì chúng có thể lặp lại