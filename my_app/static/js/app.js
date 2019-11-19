let result = document.querySelector('.result'),
img_result = document.querySelector('.img-result'),
img_w = document.querySelector('.img-w'),
img_h = document.querySelector('.img-h'),
options = document.querySelector('.options'),
save = document.querySelector('.save'),
cropped = document.querySelector('.cropped'),
dwn = document.querySelector('.download'),
upload = document.querySelector('#file-input'),
cropper = '';
ans = document.querySelector('.answer'),
process = document.querySelector('.process');


// Set constraints for the video stream
var constraints = { video: { facingMode: "user" }, audio: false };// Define constants
const cameraView = document.querySelector("#camera--view"),
    cameraOutput = document.querySelector("#camera--output"),
    cameraSensor = document.querySelector("#camera--sensor"),
    cameraTrigger = document.querySelector("#camera--trigger")// Access the device camera and stream to cameraView
function cameraStart() {
    navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function(stream) {
        track = stream.getTracks()[0];
        cameraView.srcObject = stream;
    })
    .catch(function(error) {
        console.error("Oops. Something is broken.", error);
    });
}// Take a picture when cameraTrigger is tapped
cameraTrigger.onclick = function() {
    cameraSensor.width = cameraView.videoWidth;
    cameraSensor.height = cameraView.videoHeight;
    cameraSensor.getContext("2d").drawImage(cameraView, 0, 0);
    // cameraOutput.src = cameraSensor.toDataURL("image/webp");
    // cameraOutput.classList.add("taken");
    cropped.classList.remove('hide');
    img_result.classList.remove('hide');
    let img = document.createElement('img');
    img.id = 'image'
    img.src =cameraSensor.toDataURL("image/webp")
    dwn.setAttribute('href', img.src);
    save.classList.remove('hide');
    result.appendChild(img);
    cropper = new Cropper(img);

};// Start the video stream when the window loads
window.addEventListener("load", cameraStart, false);



// save on click
save.addEventListener('click', e => {
  e.preventDefault();
  // get result to data uri
  let imgSrc = cropper.getCroppedCanvas({
    width: img_w.value // input value
  }).toDataURL();
  // remove hide class of img
  cropped.classList.remove('hide');
  img_result.classList.remove('hide');
  // show image cropped
  cropped.src = imgSrc;
  dwn.classList.remove('hide');
  ans.classList.remove('hide');
  dwn.download = 'imagename.png';
  dwn.setAttribute('href', imgSrc);
});

dwn.addEventListener('click', e => {
  e.preventDefault();
  // process.classList.remove('hide');
  url = dwn.getAttribute('href');
  // window.alert(String(url))
  console.log('landingpage?imgurl='+url);
  yobro = encodeURIComponent(url)
  // window.location.href='landingpage?imgurl='+yobro;
  window.location.href='landingpage?imgurl='+yobro+'&answer='+ans.value;
});

