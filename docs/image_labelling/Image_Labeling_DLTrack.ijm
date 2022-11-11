// Author: Paul Ritsche
// Title: Image Labeling DL_Track
// Projekt: DL_Track
// Last edited: 16.11.2021

input_dir = getDir("Choose input dir");
filelist = getFileList(input_dir);

apo_mask_dir = getDir("Choose apo mask dir");
fasc_mask_dir = getDir("Choose fasc mask dir");
image_dir = getDir("Choose image dir");

// create apo masks
function get_apo_masks(image, input_dir, image_dir, apo_mask_dir){

	// Open image
	open(input_dir + File.separator + image);
	Image_ID = getImageID();
	// Save image copy in image_dir
	selectImage(Image_ID);
	save(image_dir + File.separator + image);

	// Make mask upper Apo
	run("Select None");
	setTool("polygon");
	waitForUser("Select upper aponeurosis. Click OK when done");
	roiManager("add");
	// Make mask lower aponeurosis
	run("Select None");
	setTool("polygon");
	waitForUser("Select lower aponeurosis. Click OK when done");
	roiManager("add");
	// Combine both apos in one mask
	roiManager("combine");
	run("Create Mask");
	run("Invert");
	save(apo_mask_dir + File.separator + image);
	//Remove all selections
	roiManager("reset");
	close("*");
}

// create fascicle masks
function get_fasc_masks(image, fasc_mask_dir){
	// Open Image
	open(input_dir + File.separator + image);

	//draw fasicles
	setTool("polyline");
	waitForUser("Select fascicles. Click OK to add & draw");
	while(selectionType() >= 6) { // 6 for segmented line
		roiManager("add & draw");
		run("Select None");
		waitForUser("Select fascicles. Click OK to add & draw");
	}

	// Combine all fascicles in one mask
	roiManager("combine");
	run("Create Mask");
	run("Invert");
	save(fasc_mask_dir + File.separator + image);
	roiManager("reset");
	close("*");
}

for (i = 0; i < lengthOf(filelist); i++) {

	// Get apo masks
	get_apo_masks(filelist[i], input_dir, image_dir, apo_mask_dir);

	// Get fasc masks
	get_fasc_masks(filelist[i], fasc_mask_dir);

}
