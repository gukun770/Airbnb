$(document).ready(function(){
    console.log('document is ready');

    $('#inference').click(async function(){
        console.log('button was clicked');

        const price = parseFloat($('#price').val());
        const cleaning_fee = parseFloat($('#cleaning_fee').val());
        const bedrooms = parseFloat($('#bedrooms').val());
        const room_type = ($('#room_type').val());
        const property_type = ($('#property_type').val());

        const data = {
            price,
            cleaning_fee,
            bedrooms,
            room_type,
            property_type,
          }
        console.log(data)

        const response = await $.ajax('/inference',{
            data: JSON.stringify(data),
            method: "post",
            contentType: "application/json"
        })
        console.log(response)
        $('#revenue').val(response.prediction)


    })
})