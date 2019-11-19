let devices = await navigator.usb.getDevices();
devices.forEach(device => {
    // You can list or choose your device in this block 
});


let button = document.getElementById('request-device');
button.addEventListener('click', async () => {
  let device;
  let usbDeviceProperties = { name: "Bixolon", vendorId: 5380, productId: 31 };
  try {
    device = await navigator.usb.requestDevice({ filters: usbDeviceProperties });
  } catch (error) {
    alert('Error: ' + error.message);
  }
});

device.open()
      .then(() => device.selectConfiguration(1))
      .then(() => device.claimInterface(0));

let receivedData = await data.transferIn(1, 6);// Waiting for 6 bytes of data from endpoint #1.


navigator.usb.addEventListener('connect', event => {
    // event.device will bring the connected device
  });
  
  navigator.usb.addEventListener('disconnect', event => {
    // event.device will bring the disconnected device
  }); 
  