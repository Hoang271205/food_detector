document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const resultText = document.getElementById('result-text');
    const imagePreview = document.getElementById('image-preview');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Ngăn form gửi theo cách truyền thống

        const file = imageInput.files[0];
        if (!file) {
            resultText.textContent = 'Vui lòng chọn một ảnh!';
            return;
        }

        // Hiển thị ảnh preview
        imagePreview.innerHTML = ''; // Xóa ảnh cũ
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = document.createElement('img');
            img.src = event.target.result;
            imagePreview.appendChild(img);
        };
        reader.readAsDataURL(file);

        // Hiển thị trạng thái đang tải
        resultText.textContent = 'Đang phân tích... 🧠';
        resultText.classList.add('loading');

        // Tạo FormData để gửi ảnh
        const formData = new FormData();
        formData.append('image', file);

        try {
            // Gửi request đến server Flask
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            resultText.classList.remove('loading');

            if (response.ok) {
                // Hiển thị kết quả
                const confidence = data.confidence.toFixed(2);
                resultText.textContent = `Đây là: ${data.prediction.toUpperCase()} (Độ chính xác: ${confidence}%)`;
            } else {
                // Hiển thị lỗi từ server
                resultText.textContent = `Lỗi: ${data.error}`;
            }
        } catch (error) {
            resultText.classList.remove('loading');
            resultText.textContent = 'Lỗi: Không thể kết nối đến server.';
            console.error(error);
        }
    });
});