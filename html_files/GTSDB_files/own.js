jQuery(document).ready(function()
{
	$('#subsetselect').tooltip({
	delay: 0,
	showURL: false,
	bodyHandler: function() 
	{
		$.post("lib/subset_tooltip.php",{SUBSET: $("#selected_subset").val() },
		function(data) 
		{
			$("#tooltip div.body").html(data);
		});
		return $("#tooltip div.body").html();
	}
	}); 
});	
	function Ladebalken()
	{
			
		document.getElementById("ausgrauen").style.width ="100%";
		document.getElementById("ausgrauen").style.height ="100%";
		document.getElementById("ausgrauen").style.backgroundImage ="url(Images/transparent.png)";
		document.getElementById("ausgrauen").style.display ="block";
		document.getElementById("ausgrauen").style.position ="absolute";
		document.getElementById("ausgrauen").style.zIndex = 15;
	}
	function showsubsets()
	{
		$.post("/lib/subset.php", $("#subsetdata").serialize(),
		function(data) 
		{
			$('div#subsetinhalt').html(data);
		});
	
	}
	function showsubsets2(x)
	{
		$.post("lib/subset.php",{Untermengen: x, schreiben: "false"},
		function(data) 
		{
			$('div#subsetinhalt').html(data);
		});
	
	}
	function showsubset_option(y,x)
	{
		$.post("lib/subset.php",{Untermengen: x,option: y},
		function(data) 
		{
			$('div#subsetinhalt').html(data);
		});
	
	}
	function counter()
	{
		$.post("lib/downloadcounter.php");
	}
	function NewsMail()
	{
		tinyMCE.triggerSave();
		$.post("lib/News_Mail.php",$("#UpdateMail").serialize(),
		function(data) 
		{
			$('div#message').html(data);
		});
	}
	function NewsMail_GTSDB()
	{
		tinyMCE.triggerSave();
		$.post("lib/News_Mail_gtsdb.php",$("#UpdateMail").serialize(),
		function(data) 
		{
			$('div#message').html(data);
		});
	}	
	$(function() {
		$('#activator').click(function(){
			$("body").append('<div class="overlay" id="overlay" style="display:none;"></div><div class="box" id="box"><a class="boxclose" OnClick="CloseBox()" id="boxclose"></a><h1>Confusion matrix</h1><span class="entfernen"><center><strong>The confusion matrix is calculated</strong></center></span><br /><br /><center class="entfernen"><img src="Images/kmloader.gif" ></center><p id="Konfusionsmatrix"></p></div>');
			$('#overlay').fadeIn('fast',function(){
				$('#box').animate({'top':'60px'},300);
			});
			$.post("lib/konfusionsmatrix.php",function(data){
				$('#Konfusionsmatrix').html(data);
				$('.entfernen').remove();
			});
	
			
		});
	   
	
	});
	$(function() {
	 $('#boxclose').click(function(){
			$('#box').animate({'top':'-5000px'},300,function(){
				$('#overlay').fadeOut('fast');
				$('#box').remove();
				$('#overlay').remove();
				
			});
		});
	});
	function ShowPopUp(id,xwert,ywert)
	{
		id = 'Popup'+id.id ;
		$.post("/lib/falscherkannt.php",{ ID: "ALL", x: xwert, y: ywert }, 
		function(data) 
		{
		$('div#' + id).html(data);
		}
		);
		$(function() 
		{
			$('#' + id).dialog({ minWidth: 340 });
			$('#' + id).dialog(
			{
				autoOpen: false,
				show: "blind",
				hide: "Scale"
			});
			$('#' + id).dialog( "open" );
		});
	}
	function ShowPopUp2(id,xwert,ywert,submission)
	{
		id = 'Popup2'+id.id ;
		$.post("/lib/falscherkannt.php",{ ID: submission, x: xwert, y: ywert }, 
		function(data) 
		{
		$('div#' + id).html(data);
		}
		);
		$(function() 
		{
			$('#' + id).dialog({ minWidth: 340 });
			$('#' + id).dialog(
			{
				autoOpen: false,
				show: "blind",
				hide: "Scale"
			});
			$('#' + id).dialog( "open" );
		});
	}
	function show()
	{
		$.post("/lib/kmatrix.php", $("#submission_form").serialize(),
		function(data) 
		{
			$('div#KMatrix').html(data);
		});
	
	}
	function Histogramm()
	{
		$.post("/lib/histogramm.php", $("#submission").serialize(),
		function(data) 
		{
			$('#Histogramm').html(data);
		});
	
	}
	function showMatrix(id)
	{
		$("body").append('<div class="overlay" id="overlay""></div><div class="box" id="box" style="display:block"><a class="boxclose" OnClick="CloseBox()" id="boxclose"></a><h1>Confusion matrix</h1><span class="entfernen"><center><strong>The confusion matrix is calculated</strong></center></span><br /><br /><center class="entfernen"><img src="Images/kmloader.gif" ></center><p id="Konfusionsmatrix"></p></div>');
		$.post("/lib/Confusion_matrix.php",{ID:id},function(data)
			{
				$('#Konfusionsmatrix').html(data);
				$('.entfernen').remove();
			});
	
	
	}
	function CloseBox()
	{
		$('#box').remove();
		$('#overlay').remove();
	}
