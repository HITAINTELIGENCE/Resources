# TOÁN HỌC CƠ BẢN
*Trong đây sẽ không viết lại các phép toán mà sẽ đưa ra những định nghĩa cơ bản. Chúng mình sẽ note những trang toán học ở dưới đã viết rất xúc tích, chắt lọc và đầy đủ cho những ai quan tâm về toán cho ML.*

<!-- # Mục lục

1. [Phần 1: Đại số tuyến tính](#-Linear-Algreba-Đại-Số-Tuyến-Tính)
2. [Phần 2: Giải tích](#phần-2-tiêu-đề-2)
3. [Phần 3: Xác xuất](#)
3. [Phần 4: Thống kê](#phần-3-tiêu-đề-3) -->

# Tại sao cần học Toán?
- Luôn xuất hiện trong các buổi phỏng vấn (dù chỉ là những kiến thức cơ bản)
- 60%-70% công việc của kỹ sư AI không phải về AI
- Kỹ sư - thợ code = Tư duy + Toán

## Linear Algreba (Đại Số Tuyến Tính)
- Đại số tuyến tính là một nhánh nhỏ của toán học
- Đại số tuyến tính sẽ tập trung chủ yếu xung quanh các khái niệm về scalar, vector, matrix, tensor và các phép toán xung quanh

## Tại sao cần biết về Linear Algreba
- Hiểu được được mô hình trong Machine Learning được xây dựng và hoạt động
  - Cách mô hình tương tác với dữ liệu
  - Hàm mất mát
  - Chính quy hóa mô hình
- Dữ liệu:
  - Biểu diễn dữ liệu: table data (2d-matrix), image data (3d-tensor), video data (4d-tensor),...
  - Giảm chiều và mã hóa dữ liệu

## Các khái niệm cơ bản

### Kích thước, độ dài
- **Độ dài**: là số phần tử xuất hiện trong 1 trục của ma trận, có thể lấy độ dài bằng dùng len()
```python
    import numpy as np
    x = np.random.randint(1,10,10)
    y = [np.random.randint(1,10,10) for i in range(10)]
    print(len(x), len(y))
```
- Ở ví dụ trên, chúng ta có thể thấy là độ dài của cả hai vector trên đều là 10

- **Kích thước**: là kích thước của một ma trận, nó trả ra tập hợp độ dài của các trục trong ma trận, có thể tìm kích thước bằng thuộc tính shape
```python
    import numpy as np
    x = np.random.randint(1,10,10)
    y = np.array([np.random.randint(1,10,4) for i in range(10)])
    print(x.shape, y.shape)
```


### Scalar
- **Khái niệm**: đây là số vô hướng, chúng ta có thể bắt gặp loại số này hàng ngày như số 1,2,3
    **VD**: Bạn có 10 quả táo, thì 10 ở đây là số vô hướng

 - Ví dụ khởi tạo một số vô hướng bất kỳ bằng thư viện numpy:

 ```python
    import numpy as np
    np.random.seed(0)
    x = np.random.rand()
    y = np.random.randint(1, 10, 1)
    print(x, y)
```

- Chúng ta có thể thực hiện các phép toán cộng, trừ, nhân, chia hay là lũy thừa đối với các số này:

 ```python
    import numpy as np
    np.random.seed(0)
    x = np.random.rand()
    y = np.random.randint(1,10,1)
    print(x + y, x - y, x * y, x / y, x ** y)
```

### Vector
- **Khái niệm**: vector có thể được hiểu như là một dãy chứa các giá trị vô hướng
- Trong học máy, vector thường được dùng để biểu diễn các đại lượng đặc trưng của một vật thể
- **Ví dụ**: Chúng ta có thể biểu diễn các số ghế ngồi và chiều dài của một chiếc ô tô bằng một vector hai chiều y = (4,3), ở đây xe này có 4 ghế ngồi và dài 3 mét.

 ```python
    import numpy as np
    np.random.seed(0)
    # Khởi tạo một vector có độ dài là 10, kích thước là (10,)
    x = np.random.randint(1,10,10)
    print(x)
    print(len(x))
    print(x.shape)
```

- Trong cuộc sống thực tế, chúng ta có thể bắt gặp các khái niệm về scalar và vector
    - Tốc độ (scalar) và Vận tốc (vector)
<figure>
  <img src="https://www.mathsisfun.com/algebra/images/vector-mag-dir.svg" alt="Vector and Scalar">
  <figcaption style = "text-align: center;">Vector and Scalar</figcaption>
</figure>

### Matrix
- **Khái niệm**: Ma trận được hiểu như một mảng hình vuông hay hình chữ nhật chứa các thành phần là các giá trị vô hướng.

- Mỗi phần tử trong mảng được ký hiệu là a<sub>i</sub><sub>j</sub>, tượng trưng cho phần tử nằm tại hàng thứ i, cột thứ j.

- Trong học máy, chúng ta có thể sử dụng ma trận để biểu diễn nhiều điểm dữ liệu.
**Ví dụ**: chúng ta muốn biểu diễn các đặc trưng về chiều cao, cân nặng và điểm của 5 sinh viên, chúng ta có thể khởi tạo một ma trận có shape là 5x3, với 5 dòng tương ứng cho 5 sinh viên, mỗi sinh viên có 3 đặc trưng tương ứng.

```python
    import numpy as np
    np.random.seed(0)
    x = np.random.randint(1,10, size = (5,3))
    print(x)
    print(x.shape)
```
- Định thức của ma trận:
```python
    import numpy as np
    np.random.seed(0)
    x = np.random.randint(1,10,size = (5,5))
    y = np.linalg.det(x)
    print(y)
```
- Định thức giúp chúng ta xác định được tính khả nghịch của ma trận
- Một số ma trận đặc biệt
    - **Ma trận chuyển vị**: chúng ta có thể thu được ma trận chuyển vị bằng cách hoán đổi các dòng và cột của ma trận.
    - Trong numpy, chúng ta có thể hoán vị ma trận bằng hàm np.transpose() hoặc là sử dụng .T

     ```python
        import numpy as np
        np.random.seed(0)
        x = np.random.randint(1,10,size = (5,3))
        y = np.transpose(x)
        z = x.T
        print(y.shape, y == z, sep = '\n')
    ```

    - **Ma trận vuông**: đây là loại ma trận có số dòng bằng số cột, là loại ma trận có thể tính định thức.

    - **Ma trận đơn vị**: đây là loại ma trận đặc biệt của ma trận vuông có đường chéo chính bao gồm các số 1, còn lại là các số 0
    ```python
        import numpy as np
        np.random.seed(0)
        x = np.eye(3)
        print(x)
    ```

    - Ma trận đơn vị chính là tích của hai ma trận A và B, với B chính là ma trận nghịch đảo của A

    ```python
        import numpy as np
        np.random.seed(0)
        x = np.random.randint(1,10,size = (5,5))
        y = np.linalg.inv(x)
        print(y)
    ```
    - Điều kiện để một ma trận khả nghịch, hay có ma trận nghịch đảo là:
        1. Ma trận đó phải vuông
        2. Ma trận đó phải có định thức phải khác 0
    
    - **Ma trận tam giác**: có hai dạng ma trận tam giác, ma trận tam giác trên và dưới:
        a. Ma trận tam giác dưới:
        ```python
            import numpy as np
            np.random.seed(0)
            x = np.random.randint(1,10,size = (5,5))
            y = np.triu(x)
            print(y)
        ```
        b. Ma trận tam giác trên:
        ```python
            import numpy as np
            np.random.seed(0)
            x = np.random.randint(1,10,size = (5,5))
            y = np.tril(x)
            print(y)
        ```
    - **Ma trận đường chéo**: là kiểu ma trận có các phần tử nằm ngoài đường chéo chính đều có giá trị bằng 0
    ```python
        import numpy as np
        np.random.seed(0)
        x = np.array([1,2,3,4])
        y = np.diag(x)
        print(y)
    ```


### Tensor
- **Khái niệm**: Tensor là một cấu trúc dữ liệu đa chiều (0,1,2,...n-d)
- Thừa kế các phép toán của matrix

 ```python
    import numpy as np
    # 1-D
    x = np.array([1, 2, 3])
    # 2-D
    y = np.array([[1, 2], [3, 4]])
    # 3-D
    z = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])

  ```

**Các Phép Toán Phổ Biến**
- Các phép toán thực hiện đối với hai ma trận được giữa trên một khái niệm đó là *Vetorization*. Tham khảo thêm tại: [VIBLO](https://viblo.asia/p/python-dung-dung-vong-lap-nua-ma-dung-vectorization-y3RL1alXLao)

    - **Addition**: phép toán cộng hai ma trận
    ```python
        import numpy as np
        np.random.seed(0)
        x = np.random.randint(1,10, size = (3,3))
        y = np.random.randint(1,10, size = (3,3))
        print(x + y)
    ```
    - **Subtraction**: phép toán thực hiện trừ hai ma trận
    ```python
        import numpy as np
        np.random.seed(0)
        x = np.random.randint(1,10, size = (3,3))
        y = np.random.randint(1,10, size = (3,3))
        print(x - y)
    ```
    - **Dot Product**: thực hiện phép nhân hai ma trận, điều kiện chính là số cột của ma trận trước bằng số hàng của ma trận sau. Đối với trường hợp vector thì đây chính là tích vô hướng
    ```python
        import numpy as np
        np.random.seed(0)
        x = np.random.randint(1,10, size = (3,2))
        y = np.random.randint(1,10, size = (2,1))
        print(np.dot(x, y))
    ```

## Norm (Chuẩn)
Là một hàm số f() ánh xạ một điểm từ không gian n chiều sang tập số thực, sao cho. Tóm lại, norm để đo lường kích thước của một vector, matrix hay tensor trong không gian đa chiều.
1. f(x) ≥ 0. Dấu bằng xảy ra ⇔ x = 0.
2. f(αx) = |α|f(x), ∀α ∈ ℝ
3. f(x₁) + f(x₂) ≥ f(x₁ + x₂), ∀x₁, x₂ ∈ ℝⁿ

Công thức chung của Norm: ||x||p = (|x1|^p + |x2|^p + ... + |xn|^p) ^ (1/p) (p >= 1)

### Chuẩn của ma trận
Chuẩn  thường được dùng nhất là chuẩn Frobenius, ký hiệu là ||A||F là căn bậc hai của tổng bình phương các phần tử của ma trận.

```python
    import numpy as np
    norm1 = np.linalg.norm(vector, ord=1)  # Norm L1
    norm2 = np.linalg.norm(vector, ord=2)  # Norm L2 (Euclidean)
```

## Calculus (Giải tích)
- Là mảng trong toán học làm việc với hàm số để tìm thấy các đặc điểm của nó như là cực trị, biên, biến thiên, độ dốc,...

### Tại sao cần biết về Giải tích?
- Các mô hình machine learning có hàm mất mát (loss function) để đo lường sự sai lệch giữa giá trị dự đoán (preidcted data) và giá trị thực tế (real data). Hàm này giúp cập nhật trọng số mới của mô hình. Và giải tích giúp tối ưu hàm mất mát này.

### Đạo hàm đa chiều
Ký hiệu:
- ∇xf(x): Đạo hàm của hàm f(x) theo biến x.

```python
    import numpy as np
    import numpy as np

    x = np.array([1, 2, 4, 7, 11])

    # Xác định hàm số f(x) (ví dụ: f(x) = x^2)
    f_x = x**2

    # Tính đạo hàm của f(x)
    df_dx = np.gradient(f_x, x)
```
### Product Rule
Đạo hàm của tích 2 hàm = đạo hàm của hàm 1 * hàm 2 + hàm 1 * đạo hàm của hàm 2

### Chain Rule
Đạo hàm của hàm có đầu vào là một hàm = đạo hàm của hàm con * đạo hàm của hàm cha

## Xác xuất (Probability)
- Dùng để dự đoán xảy ra của một sự kiện trong tương lai
- Mô hình trong ML đưa ra các xác xuất xảy ra của sự kiện dựa vào đầu vào.

### Xác xuất thực nghiệm
- Xác suất thực nghiệm là xác suất được đo lường hoặc ước tính dựa trên dữ liệu quan sát được từ thực tế.
- Đây là xác suất dựa trên kinh nghiệm hoặc thực tiễn, được tính bằng cách đếm số lần một sự kiện xảy ra trong một số lần thử nghiệm hoặc quan sát.


### Xác xuất lý thuyết
- Xác suất lý thuyết là xác suất được tính toán dựa trên một lý thuyết xác suất cụ thể.
- Đây là xác suất được dự đoán dựa trên một mô hình toán học hoặc giả định về cách mà sự kiện có thể xảy ra.


### Xác xuất kết hợp (Joint Probability)
- Xác suất kết hợp của hai sự kiện là xác suất cả hai sự kiện xảy ra cùng một lúc.
- Nó thường được ký hiệu là P(A∩B) và được tính bằng cách nhân xác suất của hai sự kiện riêng biệt P(A) và P(B), nếu sự kiện A và B độc lập, hoặc bằng P(A∣B)×P(B) nếu không độc lập.


### Xác xuất điều kiện (Conditional Probability)
- Xác suất điều kiện của sự kiện A trong điều kiện B là xác suất mà sự kiện A xảy ra dưới điều kiện rằng sự kiện B đã xảy ra.
- Nó được ký hiệu là P(A∣B) và được tính bằng cách chia xác suất kết hợp P(A∩B) cho xác suất của sự kiện điều kiện P(B).


## Thống kê (Statistical)
- Phân tích và đánh giá sự kiện đã xảy ra trong quá khứ
- Phân tích các sự kiện đã xảy ra để đưa dữ liệu vào mô hình toán học cho phù hợp
- Có thể sử dụng các biểu đồ để thống kê dữ liệu và các công cụ giá phân phối, giá

### Thống kê mô tả
- Làm việc trên tập mẫu
    
- Xu hướng tập trung của dữ liệu, đo lường biến động của dữ liệu
    
### Thống kê suy luận
- Làm việc trên tập con của mẫu

### Kỳ vọng
> Giá trị mong đợi của biến

### Phương sai
> Đo lường mức độ biến động của biến ngẫu nhiên xung quanh kỳ vọng của nó.

### Các phân phối thường gặp
- **Phân phối Bernoulli:** Phân phối nhị phân, biểu diễn cho outcome 2 lớp

- **Phân phối Categorical:** Phân phối biểu diễn cho outcome nhiều lớp

- **Phân phối chuẩn (Norm):** Phân phối biểu diễn cho giá trị liên tục

- **Phân phối Beta (Beta distributed):** Phân phối biểu diễn biến động cho phân phối Bernoulli

- **Phân phối Dirichlet:** phân phối biểu diễn biến động cho phân phối Categorical

# Nguồn có thể tự học thêm kiến thức về toán cho ML
- [Ebook - Machine Learning | Vũ Tiệp](https://github.com/tiepvupsu/ebookMLCB)
- [Machine Learning cơ bản](https://machinelearningcoban.com/math)
- [Standford - CS229](https://stanford.edu/~shervine/l/vi/teaching/cs-229)

