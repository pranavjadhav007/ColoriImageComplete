$(document).ready(function() {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#imagePreview1').hide();
    $('#btn-predict').hide();


    function readURL(input, previewElement) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                previewElement.css('background-image', 'url(' + e.target.result + ')');
                previewElement.hide();
                previewElement.fadeIn(650);
                $('#btn-predict').show();
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload1").change(function() {
        $('.image-section').show();
        $('#btn-predict').hide();
        $('#result').hide();
        readURL(this, $('#imagePreview1'));
        $('#imagePreview1').show();
    });

    $('#btn-predict').click(function() {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                var finalMaskImageURL = data.final_req_image;
                console.log(finalMaskImageURL)
                $('.loader').hide();
                $('#result').show();
                $("#final_mask_image").attr('src', finalMaskImageURL);
                $("#final_mask_image").show();
            }
        });
    });
});