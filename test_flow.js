const { chromium } = require('C:/Users/Abhinai/AppData/Roaming/npm/node_modules/@playwright/cli/node_modules/playwright-core');

(async () => {
    try {
        const browser = await chromium.launch({ channel: 'msedge', headless: true });
        const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
        const page = await context.newPage();

        console.log('Navigating to http://127.0.0.1:5000...');
        await page.goto('http://127.0.0.1:5000');
        await page.screenshot({ path: 'screenshot_01_upload.png', fullPage: true });
        console.log('Saved screenshot_01_upload.png');

        console.log('Clicking sample artwork button...');
        await page.click('#sample-art-btn');
        await page.waitForSelector('#preview-section:not(.hidden)', { timeout: 5000 });
        await page.screenshot({ path: 'screenshot_02_preview.png', fullPage: true });
        console.log('Saved screenshot_02_preview.png');

        console.log('Clicking Execute Restoration button...');
        await page.click('#restore-btn');
        
        // Take immediate screenshot of processing state
        await page.waitForTimeout(600);
        await page.screenshot({ path: 'screenshot_03_processing.png', fullPage: true });
        console.log('Saved screenshot_03_processing.png');

        console.log('Waiting for restoration result...');
        await page.waitForSelector('#result-section:not(.hidden)', { timeout: 45000 });
        await page.screenshot({ path: 'screenshot_04_result_slider.png', fullPage: true });
        console.log('Saved screenshot_04_result_slider.png');

        console.log('Testing side-by-side mode toggle...');
        await page.click('#mode-side-btn');
        await page.screenshot({ path: 'screenshot_05_result_side.png', fullPage: true });
        console.log('Saved screenshot_05_result_side.png');

        // Test mobile layout
        console.log('Testing mobile layout...');
        await page.setViewportSize({ width: 390, height: 844 });
        await page.screenshot({ path: 'screenshot_06_mobile.png', fullPage: true });
        console.log('Saved screenshot_06_mobile.png');

        await browser.close();
        console.log('PLAYWRIGHT_FLOW_SUCCESS');
    } catch (err) {
        console.error('PLAYWRIGHT_FLOW_ERROR:', err);
        process.exit(1);
    }
})();
