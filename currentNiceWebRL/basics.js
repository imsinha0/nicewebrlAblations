async function isFullscreen() {
  try {
    return document.fullscreenElement != null;
  } catch (error) {
    console.warn('Fullscreen check failed:', error);
    return false;  // Return a safe default
  }
} 

async function getImageSeenTime() {
  return window.imageSeenTime;
}

async function getSystemReadyTime() {
  return window.systemReadyTime;
} 

// Function to send ping to server every 30 seconds
async function pingServer() {
  console.log('Starting ping loop');
  while (true) {
    try {
      const message = `Ping ${Math.floor(Math.random() * 10000)}`;
      // Await the asynchronous emitEvent call
      await emitEvent('ping', { message: message });
      console.log(`Ping sent: ${message}`);
    } catch (err) {
      console.error('Error pinging server:', err);
    }
    // Wait for 30 seconds before sending the next ping
    seconds = 30
    try {
      await new Promise(resolve => setTimeout(resolve, seconds * 1000));
    } catch (err) {
      console.error('Error in ping loop:', err);
    }
  }
}

// Function to toggle spacebar behavior
let spacebarPrevented = false; // Default to preventing spacebar
function preventDefaultSpacebarBehavior(shouldPrevent) {
  spacebarPrevented = shouldPrevent;
  return spacebarPrevented;
}

document.addEventListener('DOMContentLoaded', async function () {

  ////////////////
  // Wait for NiceGUI to be fully ready, then signal to server
  ////////////////
  function waitForNiceGuiReady() {
    const focusableCard = document.querySelector('[tabindex="0"]');
    if (focusableCard) {
      focusableCard.focus();
      window.niceGuiReady = true;
    } else {
      setTimeout(waitForNiceGuiReady, 50);
    }
  }
  
  // Initialize the ready flag and start checking
  window.niceGuiReady = false;
  setTimeout(waitForNiceGuiReady, 50);

  ////////////////
  // Start pinging the server once the DOM content is fully loaded
  ////////////////
  pingServer();

  ////////////////
  // remove default behavior
  ////////////////
  window.debug = 0;
  window.require_fullscreen = false;
  window.accept_keys = false;
  window.next_states = null;
  window.key_count = 0;
  window.systemReadyTime = null;

  ////////////////
  // Monitor when system becomes ready for next input
  ////////////////
  // function monitorSystemReady() {
  //   // Use a simple polling approach to detect when accept_keys becomes true
  //   const checkReady = () => {
  //     if (window.accept_keys && window.systemReadyTime === null) {
  //       window.systemReadyTime = new Date();
  //       console.log('System ready for next input at:', window.systemReadyTime);
  //     }
  //   };
    
  //   // Check every 10ms for responsiveness
  //   setInterval(checkReady, 10);
  // }
  
  // monitorSystemReady();
  
  ////////////////
  // Track when system is actually ready for next input after processing
  ////////////////
  // function setSystemReadyForNextInput() {
  //   window.systemReadyTime = new Date();
  //   console.log('System ready for next input at:', window.systemReadyTime);
  // }
  
  // // Make this function globally available
  // window.setSystemReadyForNextInput = setSystemReadyForNextInput;

  ////////////////
  // how to handle key presses?
  ////////////////
  document.addEventListener('keydown', async function(event) {
    
    // Skip if the chat input is focused
    console.log('-------------------------------------------')
    console.log('active element:', document.activeElement.id);
    if (document.activeElement && document.activeElement.id === 'chat-input') {
      //console.log('chat input focused');
      return;
    }

    // Check if the key pressed is spacebar
    if ((event.key === " " || event.code === "Space") && spacebarPrevented) {
      // Prevent the default action (toggling fullscreen)
      event.preventDefault();
      if (window.key_count < 1) {console.log('prevented spacebar');}
    }

    // Prevent default behavior for arrow keys
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
      if (window.key_count < 1) { console.log('preventing default arrow key behavior'); }
      event.preventDefault();
    }

    console.log(event.key);
    if (window.next_states !== null && window.accept_keys && event.key in window.next_states) {
      //console.log('registering key press');
      if (!window.require_fullscreen || await isFullscreen()) {
        // Record the current time when the keydown event occurs FIRST
        var keydownTime = new Date();
        

        await emitEvent('key_pressed', {
          key: event.key,
          keydownTime: keydownTime,
          imageSeenTime: window.imageSeenTime,
          systemReadyTime: window.systemReadyTime
        });
        console.log('emitted key_pressed');

        next_state = window.next_states[event.key];
        window.next_states = null;
        var imgElement = document.getElementById('stateImage');
        if (imgElement !== null) {
          imgElement.src = next_state;
          // Set imageSeenTime when the image updates to the next image
          window.imageSeenTime = new Date();
        }

      }
    }
    window.key_count = window.key_count + 1;

  }, true); // Using capturing phase to catch the event before other handlers


})