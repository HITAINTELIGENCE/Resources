# GRADIENT DESCENT

## Ôn tập các khái niệm cơ bản

![](/Gradient_Descent/gradient_descent.png)

- Cực trị của hàm số là giá trị khiến cho hàm số đổi chiều khi biến thiên.
- Điểm cực tiểu là một điểm trên đồ thị của một hàm số mà giá trị của hàm số tại điểm đó nhỏ nhất so với các giá trị xung quanh. Trong học máy, chúng ta sẽ sử dụng thuật ngữ local minimum thay thế cho "điểm cực tiểu".
- Điểm có giá trị nhỏ nhất là một local minimum đặc biệt, và được gọi là global minimum.
- Điểm local minimum sẽ khiến cho đạo hàm của hàm số  bằng 0, điểm global minimum sẽ khiến cho hàm số đạt giá trị nhỏ nhất.

- Đối với hình trên, chúng 

## Motivations
- Việc tìm  weight sao cho loss function là minimum có thể được thực hiện theo hai cách phổ biến như sau:
    - Tìm nghiệm của đạo hàm, nhưng đa số cách này trở nên bất khả thi trong đa số các trường hợp vì đạo hàm có thể có rất nhiều nghiệm (local minimum) nên việc tìm ra giá trị nhỏ nhất (global minimum) sẽ rất khó khăn, một lý do khác là đạo hàm có thể vô nghiệm hoặc rất phức tạp để tính toán.
    -  