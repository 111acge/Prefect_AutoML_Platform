import { createI18n } from 'vue-i18n'
import zhCN from './locales/zh-CN.js'
import en from './locales/en.js'

const messages = {
  'zh-CN': zhCN,
  en,
}

const savedLocale = localStorage.getItem('locale') || 'zh-CN'
const defaultLocale = messages[savedLocale] ? savedLocale : 'zh-CN'

const i18n = createI18n({
  legacy: false,
  locale: defaultLocale,
  fallbackLocale: 'zh-CN',
  messages,
  missing: (locale, key) => {
    // 开发环境下方便定位缺失键；生产环境可返回空字符串
    if (import.meta.env.DEV) {
      console.warn(`[i18n] Missing translation: ${locale}.${key}`)
    }
    return key
  },
})

export const SUPPORTED_LOCALES = [
  { value: 'zh-CN', label: '中文' },
  { value: 'en', label: 'English' },
]

export function setLocale(locale) {
  if (!messages[locale]) {
    locale = 'zh-CN'
  }
  localStorage.setItem('locale', locale)
  i18n.global.locale.value = locale
}

export function getLocale() {
  return i18n.global.locale.value
}

export default i18n
