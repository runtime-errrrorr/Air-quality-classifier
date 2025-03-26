

function goToLocation() {
    let location = document.getElementById("location").value;
    if (location) {
        window.location.href = location;
    } else {
        alert("Please select a location first!");
    }
}